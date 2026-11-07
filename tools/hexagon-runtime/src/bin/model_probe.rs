//! Externally supervised, terminal original-Swift graph qualification worker.
//!
//! The Runner owner verifies source/package/driver identities and creates this
//! worker suspended in its deadline-owned, kill-on-close Job. This executable
//! grants no service admission. Up to eight real graph tokens offload only the original
//! first FFN up projection. Complete CPU logits and logical state must match
//! before publication. Streaming controls independently replay every returned
//! integer; the explicit resident timing screen uses graph parity alone.
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
    cell::Cell,
    collections::{BTreeMap, BTreeSet},
    env,
    fs::{self, File},
    io::{self, Read as _, Write as _},
    path::{Component, Path, PathBuf},
    process::ExitCode,
    time::{Duration, Instant},
};

use bonsai_hexagon_runtime::{
    GuardedQ2Executor, HmxQ2Executor, NativeQ2Executor, PlacedProjectionExecutor,
};
use bonsai_prism_runtime::{
    Decoder, Result,
    graph::{Qwen35, StateComponent},
    kernels::{MatvecBackend, Pq2ProjectionExecutor, RowParallelism},
    model::{Model, lora::RankOneAdapter, swift},
};
use hexagon_native_rpc::{
    Driver, DriverLocation, Error, GuardedQ2Submitter, HmxQ2ProjectionLimits, HmxQ2Submitter,
    HmxQ2WeightHandle, NativeQ2ProjectionLimits, NativeQ2Submitter, NativeQ2WeightHandle,
    Q2_PROJECTION_MAX_OPERATIONS, Q2ProjectionLimits,
};
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest as _, Sha256};

const PROJECTION: &str = "blk.0.ffn_up.weight";
const MAX_REQUEST_BYTES: u64 = 8192;
const MODEL_MEMORY_LIMIT: usize = 12 * 1024 * 1024 * 1024;
const MAX_HOST_PAYLOAD: usize = 64 * 1024 * 1024;
const MAX_TOKENS: usize = 8;
const RESIDENT_WEIGHT_BYTES: usize = 17_408 * 36 * 40;
const RESIDENT_METADATA_BYTES: usize = 64 * 1024;
const NATIVE_Q2_RESIDENT_METADATA_BYTES: usize = 512 * 1024;
// Exact original-Swift geometry is independently enforced by swift::load:
// 48 recurrent blocks, state head128, value heads48, key groups16 and conv4.
// visit_state promotes the sixteen full-attention blocks' F16 K/V to F32.
const CONTROL_RECURRENT_BYTES: usize = 48 * (128 * 128 * 48 + 3 * (2 * 128 * 16 + 128 * 48)) * 4;
const CONTROL_KV_BYTES_PER_TOKEN: usize = 16 * 2 * 256 * 4 * 4;

#[derive(Debug)]
struct ControlStateBudgetExceeded {
    required: usize,
    available: usize,
}

impl std::fmt::Display for ControlStateBudgetExceeded {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "CPU control snapshots require {} bytes, exceeding the {} byte allowance",
            self.required, self.available
        )
    }
}

impl std::error::Error for ControlStateBudgetExceeded {}

fn control_state_bytes(tokens: usize) -> Result<usize> {
    require((1..=MAX_TOKENS).contains(&tokens))?;
    Ok(tokens * CONTROL_RECURRENT_BYTES + CONTROL_KV_BYTES_PER_TOKEN * tokens * (tokens + 1) / 2)
}

#[derive(Clone, Copy, Default, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "kebab-case")]
enum AdapterCase {
    #[default]
    None,
    LoadedZero,
    #[serde(rename = "standard-scale-1.3")]
    StandardScale13,
}

impl AdapterCase {
    const fn name(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::LoadedZero => "loaded-zero",
            Self::StandardScale13 => "standard-scale-1.3",
        }
    }

    const fn loaded(self) -> bool {
        !matches!(self, Self::None)
    }

    const fn scale(self) -> f32 {
        if matches!(self, Self::StandardScale13) {
            1.3
        } else {
            0.0
        }
    }
}

#[derive(Clone, Copy, Default, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "kebab-case")]
enum CpuBackend {
    #[default]
    Scalar,
    Neon,
}

impl CpuBackend {
    const fn name(self) -> &'static str {
        match self {
            Self::Scalar => "scalar",
            Self::Neon => "neon",
        }
    }

    const fn backend(self) -> MatvecBackend {
        match self {
            Self::Scalar => MatvecBackend::Scalar,
            Self::Neon => MatvecBackend::ExperimentalNeon,
        }
    }

    const fn available(self) -> bool {
        matches!(self, Self::Scalar)
            || cfg!(all(
                feature = "experimental-neon",
                target_os = "windows",
                target_arch = "aarch64",
                target_env = "msvc",
                target_feature = "neon"
            ))
    }
}

const fn default_row_workers() -> usize {
    1
}

#[derive(Clone, Copy, Default, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "kebab-case")]
enum ExecutionBackend {
    #[default]
    GuardedHvx,
    HmxBitplane,
    NativeQ2,
}

impl ExecutionBackend {
    const fn name(self) -> &'static str {
        match self {
            Self::GuardedHvx => "guarded-hvx",
            Self::HmxBitplane => "hmx-bitplane",
            Self::NativeQ2 => "native-q2",
        }
    }

    const fn operations_per_projection(self) -> u32 {
        match self {
            Self::GuardedHvx | Self::HmxBitplane => 40,
            Self::NativeQ2 => 1,
        }
    }

    const fn values_per_token(self) -> u64 {
        match self {
            Self::GuardedHvx => 1_392_640,
            Self::HmxBitplane => 5_570_560,
            Self::NativeQ2 => 0,
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
    projections: Option<Vec<String>>,
    #[serde(default)]
    execution_backend: ExecutionBackend,
    #[serde(default)]
    resident_host_weights: bool,
    #[serde(default)]
    batch_four_operations: bool,
    #[serde(default)]
    batch_ten_operations: bool,
    #[serde(default)]
    adapter_path: Option<PathBuf>,
    #[serde(default)]
    adapter_case: AdapterCase,
    #[serde(default)]
    cpu_backend: CpuBackend,
    #[serde(default = "default_row_workers")]
    row_workers: usize,
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

fn projection_dimensions(name: &str, backend: ExecutionBackend) -> Option<(usize, usize)> {
    let rest = name.strip_prefix("blk.")?;
    let (index, suffix) = rest.split_once('.')?;
    if index.is_empty() || index.len() > 2 || !index.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    let layer = index.parse::<usize>().ok().filter(|&layer| layer < 64)?;
    let full_attention = (layer + 1).is_multiple_of(4);
    match backend {
        ExecutionBackend::GuardedHvx | ExecutionBackend::HmxBitplane => {
            matches!(suffix, "ffn_up.weight" | "ffn_gate.weight").then_some((5120, 17_408))
        }
        ExecutionBackend::NativeQ2 => match suffix {
            "ffn_up.weight" | "ffn_gate.weight" => Some((5120, 17_408)),
            "ffn_down.weight" => Some((17_408, 5120)),
            "attn_qkv.weight" if !full_attention => Some((5120, 10_240)),
            "attn_gate.weight" if !full_attention => Some((5120, 6144)),
            "ssm_out.weight" if !full_attention => Some((6144, 5120)),
            "attn_q.weight" if full_attention => Some((5120, 12_288)),
            "attn_k.weight" | "attn_v.weight" if full_attention => Some((5120, 1024)),
            "attn_output.weight" if full_attention => Some((6144, 5120)),
            _ => None,
        },
    }
}

fn expected_resident_weight_bytes(
    projections: &[&str],
    backend: ExecutionBackend,
) -> Result<usize> {
    let mut total = 0_usize;
    for &name in projections {
        let (columns, rows) = projection_dimensions(name, backend).ok_or_else(reject)?;
        let bytes = (rows / 32)
            .checked_mul(columns / 64)
            .and_then(|tiles| tiles.checked_mul(576))
            .ok_or_else(reject)?;
        total = total.checked_add(bytes).ok_or_else(reject)?;
    }
    Ok(total)
}

impl Request {
    fn tokens(&self) -> &[u32] {
        self.token_ids
            .as_deref()
            .unwrap_or_else(|| std::slice::from_ref(&self.token_id))
    }

    fn projection_names(&self) -> Vec<&str> {
        self.projections.as_ref().map_or_else(
            || vec![PROJECTION],
            |names| names.iter().map(String::as_str).collect(),
        )
    }

    fn batch_operations(&self) -> Result<u32> {
        require(!(self.batch_four_operations && self.batch_ten_operations))?;
        Ok(if self.batch_ten_operations {
            10
        } else if self.batch_four_operations {
            4
        } else {
            1
        })
    }

    fn limits(&self) -> Result<Q2ProjectionLimits> {
        let tokens = self.tokens();
        let projections = self.projection_names();
        let max_projections = match self.execution_backend {
            ExecutionBackend::NativeQ2 => {
                bonsai_hexagon_runtime::DEFAULT_NATIVE_Q2_PROJECTIONS.len()
            }
            ExecutionBackend::GuardedHvx | ExecutionBackend::HmxBitplane => {
                if self.resident_host_weights { 2 } else { 4 }
            }
        };
        let unique_projections: BTreeSet<&str> = projections.iter().copied().collect();
        let batch_ops = self.batch_operations()?;
        let resident_bytes = expected_resident_weight_bytes(&projections, self.execution_backend)
            .unwrap_or(usize::MAX);
        require(
            self.schema_version == 1
                && (1..=MAX_TOKENS).contains(&tokens.len())
                && tokens.first() == Some(&self.token_id)
                && tokens.iter().all(|&token| {
                    usize::try_from(token).is_ok_and(|token| token < swift::VOCAB_SIZE)
                })
                && (1..=max_projections).contains(&projections.len())
                && projections.first() == Some(&PROJECTION)
                && unique_projections.len() == projections.len()
                && projections
                    .iter()
                    .all(|&name| projection_dimensions(name, self.execution_backend).is_some())
                && (1..=16).contains(&self.context_limit)
                && self.context_limit >= tokens.len()
                && (swift::TENSOR_PAYLOAD_BYTES..=MODEL_MEMORY_LIMIT)
                    .contains(&self.model_memory_limit_bytes)
                && (1..=bonsai_prism_runtime::graph::MAX_STATE_BYTES)
                    .contains(&self.state_budget_bytes)
                && (1..=MAX_HOST_PAYLOAD).contains(&self.maximum_owned_payload_bytes)
                && self.maximum_operations
                    >= u32::try_from(tokens.len() * projections.len())?
                        * self.execution_backend.operations_per_projection()
                && (!self.resident_host_weights
                    || match self.execution_backend {
                        ExecutionBackend::HmxBitplane => {
                            resident_bytes <= 2 * RESIDENT_WEIGHT_BYTES
                        }
                        ExecutionBackend::NativeQ2 => {
                            resident_bytes
                                <= hexagon_native_rpc::NATIVE_Q2_MAX_RESIDENT_WEIGHT_BYTES
                        }
                        ExecutionBackend::GuardedHvx => false,
                    })
                && (batch_ops == 1
                    || (self.execution_backend == ExecutionBackend::HmxBitplane
                        && self.maximum_operations.is_multiple_of(batch_ops)))
                && self.adapter_path.is_some() == self.adapter_case.loaded()
                && self.cpu_backend.available()
                && matches!(self.row_workers, 1 | 4)
                && (self.row_workers == 1 || cfg!(feature = "parallel-rows"))
                && self.expected_parent_pid != 0
                && self.expected_parent_creation_time != 0,
        )?;
        let required = control_state_bytes(tokens.len())?;
        if required > self.state_budget_bytes {
            return Err(ControlStateBudgetExceeded {
                required,
                available: self.state_budget_bytes,
            }
            .into());
        }
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
        if let Some(path) = &self.adapter_path {
            external(path)?;
        }
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
    evaluation_duration: Duration,
}

fn graph_for_arm(model: Model, request: &Request, cpu_baseline: bool) -> Result<Qwen35> {
    // The loaded-zero case deliberately uses a bare CPU graph as the oracle:
    // equal results prove identity with the original base, not just agreement
    // between two copies of the same loaded adapter path.
    let load_adapter = request.adapter_case.loaded()
        && !(cpu_baseline && request.adapter_case == AdapterCase::LoadedZero);
    let mut graph = if load_adapter {
        let path = request.adapter_path.as_deref().ok_or_else(reject)?;
        let adapter = RankOneAdapter::load_swift(path, &model)?;
        Qwen35::with_rank_one_adapter(
            model,
            request.context_limit,
            request.state_budget_bytes,
            request.cpu_backend.backend(),
            adapter,
            request.adapter_case.scale(),
        )?
    } else {
        Qwen35::with_state_budget(
            model,
            request.context_limit,
            request.state_budget_bytes,
            request.cpu_backend.backend(),
        )?
    };
    graph.set_row_parallelism(RowParallelism::new(request.row_workers)?);
    require(graph.row_workers() == request.row_workers)?;
    Ok(graph)
}

fn cpu_stage<T>(
    stage: &str,
    token_index: Option<usize>,
    work: impl FnOnce() -> Result<T>,
) -> Result<T> {
    cpu_stage_with_report(stage, token_index, work, emit)
}

fn cpu_stage_with_report<T>(
    stage: &str,
    token_index: Option<usize>,
    work: impl FnOnce() -> Result<T>,
    report: impl FnOnce(&Value) -> Result<()>,
) -> Result<T> {
    let result = work();
    if result.is_err() {
        report(&json!({"event":"cpu-control-failure","stage":stage,"tokenIndex":token_index}))?;
    }
    result
}

fn cpu_control(model: Model, request: &Request) -> Result<Vec<Control>> {
    let mut graph = cpu_stage("graph-construction", None, || {
        graph_for_arm(model, request, true)
    })?;
    let mut controls = Vec::new();
    let mut remaining_state_bytes = request.state_budget_bytes;
    controls.try_reserve_exact(request.tokens().len())?;
    for (index, &token) in request.tokens().iter().enumerate() {
        let started = Instant::now();
        let logits = cpu_stage("evaluate", Some(index), || graph.evaluate(token))?;
        let evaluation_duration = started.elapsed();
        let rows = cpu_stage("state-capture", Some(index), || {
            capture_state(&graph, remaining_state_bytes)
        })?;
        for row in &rows {
            remaining_state_bytes = remaining_state_bytes
                .checked_sub(row.bits.len().checked_mul(4).ok_or_else(reject)?)
                .ok_or_else(reject)?;
        }
        controls.push(Control {
            logits,
            state_sha256: cpu_stage("state-digest", Some(index), || state_digest(&graph))?,
            rows,
            evaluation_duration,
        });
    }
    Ok(controls)
}

#[derive(Default)]
struct SubmitTimings {
    native: Cell<Duration>,
    oracle: Cell<Duration>,
}

fn timed<T>(accumulator: &Cell<Duration>, work: impl FnOnce() -> T) -> T {
    let started = Instant::now();
    let result = work();
    accumulator.set(accumulator.get().saturating_add(started.elapsed()));
    result
}

fn token_timings(
    cpu: Duration,
    native_graph: Duration,
    native_submit: Duration,
    integer_oracle: Duration,
) -> Value {
    json!({
        "cpuEvaluationMilliseconds":cpu.as_secs_f64() * 1000.0,
        "nativeCheckedEvaluationMilliseconds":native_graph.as_secs_f64() * 1000.0,
        "nativeSubmitAndWaitMilliseconds":native_submit.as_secs_f64() * 1000.0,
        "independentIntegerOracleMilliseconds":integer_oracle.as_secs_f64() * 1000.0,
        "nativeEvaluationExcludingIntegerOracleMilliseconds":
            native_graph.saturating_sub(integer_oracle).as_secs_f64() * 1000.0,
    })
}

/// Independent scalar oracle over each signed operation's serialized operands.
struct CheckedSubmitter<'a> {
    native: &'a mut dyn GuardedQ2Submitter,
    timings: &'a SubmitTimings,
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
        timed(&self.timings.native, || {
            self.native.submit(weights, activation, output)
        })?;
        let count = timed(&self.timings.oracle, || {
            compare_limbs(weights, activation, output)
        })
        .map_err(|_| Error::Contract)?;
        self.compared = self.compared.checked_add(count).ok_or(Error::Contract)?;
        self.failed = false;
        Ok(())
    }
}

/// Independent decoding of the serialized signed-kernel Q2 tiles and all eight
/// binary activation planes. This does not reuse the production HMX packer or
/// reconstruction helpers.
#[allow(
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
            let mut dot = 0_i16;
            for (lane, &value) in values.iter().enumerate() {
                if value == 1.0 {
                    let local = lane % 64;
                    let tile = (row / 32 * 2 + lane / 64) * 576;
                    let byte = weights[tile + local / 16 * 128 + row % 32 * 4 + local % 4];
                    let symbol = (byte >> (local / 4 % 4 * 2)) & 3;
                    dot += i16::from(symbol) - 1;
                }
            }
            require(output[plane * rows + row] == f32::from(dot))?;
        }
    }
    Ok(u64::try_from(output.len())?)
}

struct CheckedHmxSubmitter<'a> {
    native: &'a mut dyn HmxQ2Submitter,
    timings: &'a SubmitTimings,
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
            timed(&self.timings.native, || {
                self.native.submit(weights, activation, output)
            })?;
            let count = timed(&self.timings.oracle, || {
                compare_planes(weights, activation, output)
            })
            .map_err(|_| Error::Contract)?;
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

    fn submit_batch4(
        &mut self,
        weights: [&[u8]; 4],
        activations: [&[[f32; 128]; 8]; 4],
        outputs: [&mut [f32]; 4],
    ) -> std::result::Result<(), Error> {
        if self.failed {
            for output in outputs {
                output.fill(f32::NAN);
            }
            return Err(Error::Contract);
        }
        self.failed = true;
        let [out0, out1, out2, out3] = outputs;
        let result = (|| {
            timed(&self.timings.native, || {
                self.native
                    .submit_batch4(weights, activations, [out0, out1, out2, out3])
            })?;
            let count = timed(&self.timings.oracle, || -> Result<u64> {
                let mut total = 0_u64;
                for ((weight, activation), output) in weights
                    .into_iter()
                    .zip(activations)
                    .zip([&*out0, &*out1, &*out2, &*out3])
                {
                    let member_count = compare_planes(weight, activation, output)?;
                    total = total.checked_add(member_count).ok_or_else(reject)?;
                }
                Ok(total)
            })
            .map_err(|_| Error::Contract)?;
            self.compared = self.compared.checked_add(count).ok_or(Error::Contract)?;
            Ok(())
        })();
        if result.is_ok() {
            self.failed = false;
        } else {
            for output in [out0, out1, out2, out3] {
                output.fill(f32::NAN);
            }
        }
        result
    }

    fn submit_batch10(
        &mut self,
        weights: [&[u8]; 10],
        activations: [&[[f32; 128]; 8]; 10],
        outputs: [&mut [f32]; 10],
    ) -> std::result::Result<(), Error> {
        if self.failed {
            for output in outputs {
                output.fill(f32::NAN);
            }
            return Err(Error::Contract);
        }
        self.failed = true;
        let [out0, out1, out2, out3, out4, out5, out6, out7, out8, out9] = outputs;
        let result = (|| {
            timed(&self.timings.native, || {
                self.native.submit_batch10(
                    weights,
                    activations,
                    [out0, out1, out2, out3, out4, out5, out6, out7, out8, out9],
                )
            })?;
            let count = timed(&self.timings.oracle, || -> Result<u64> {
                let mut total = 0_u64;
                for ((weight, activation), output) in weights.into_iter().zip(activations).zip([
                    &*out0, &*out1, &*out2, &*out3, &*out4, &*out5, &*out6, &*out7, &*out8, &*out9,
                ]) {
                    let member_count = compare_planes(weight, activation, output)?;
                    total = total.checked_add(member_count).ok_or_else(reject)?;
                }
                Ok(total)
            })
            .map_err(|_| Error::Contract)?;
            self.compared = self.compared.checked_add(count).ok_or(Error::Contract)?;
            Ok(())
        })();
        if result.is_ok() {
            self.failed = false;
        } else {
            for output in [out0, out1, out2, out3, out4, out5, out6, out7, out8, out9] {
                output.fill(f32::NAN);
            }
        }
        result
    }
}

/// Resident timing mode checks complete graph parity instead of replaying dots.
/// It owns no copied weight oracle and refuses any streaming fallback.
struct TimedHmxSubmitter<'a> {
    native: &'a mut dyn HmxQ2Submitter,
    timings: &'a SubmitTimings,
    failed: bool,
}

impl HmxQ2Submitter for TimedHmxSubmitter<'_> {
    fn remaining_operations(&self) -> Option<u64> {
        if self.failed {
            Some(0)
        } else {
            self.native.remaining_operations()
        }
    }

    fn residency_capacity_bytes(&self) -> usize {
        if self.failed {
            0
        } else {
            self.native.residency_capacity_bytes()
        }
    }

    fn prepare_weights(
        &mut self,
        weights: &[u8],
    ) -> std::result::Result<Option<HmxQ2WeightHandle>, Error> {
        if self.failed {
            return Err(Error::Contract);
        }
        self.failed = true;
        let handle = self.native.prepare_weights(weights)?;
        if handle.is_some() {
            self.failed = false;
        }
        Ok(handle)
    }

    fn submit_resident(
        &mut self,
        handle: &HmxQ2WeightHandle,
        activation: &[[f32; 128]; 8],
        output: &mut [f32],
    ) -> std::result::Result<(), Error> {
        if self.failed {
            output.fill(f32::NAN);
            return Err(Error::Contract);
        }
        self.failed = true;
        let result = timed(&self.timings.native, || {
            self.native.submit_resident(handle, activation, output)
        });
        if result.is_ok() {
            self.failed = false;
        } else {
            output.fill(f32::NAN);
        }
        result
    }

    fn submit_resident_batch4(
        &mut self,
        handles: [&HmxQ2WeightHandle; 4],
        activations: [&[[f32; 128]; 8]; 4],
        outputs: [&mut [f32]; 4],
    ) -> std::result::Result<(), Error> {
        if self.failed {
            for output in outputs {
                output.fill(f32::NAN);
            }
            return Err(Error::Contract);
        }
        self.failed = true;
        let [out0, out1, out2, out3] = outputs;
        let result = timed(&self.timings.native, || {
            self.native
                .submit_resident_batch4(handles, activations, [out0, out1, out2, out3])
        });
        if result.is_ok() {
            self.failed = false;
        } else {
            for output in [out0, out1, out2, out3] {
                output.fill(f32::NAN);
            }
        }
        result
    }

    fn submit_resident_batch10(
        &mut self,
        handles: [&HmxQ2WeightHandle; 10],
        activations: [&[[f32; 128]; 8]; 10],
        outputs: [&mut [f32]; 10],
    ) -> std::result::Result<(), Error> {
        if self.failed {
            for output in outputs {
                output.fill(f32::NAN);
            }
            return Err(Error::Contract);
        }
        self.failed = true;
        let [out0, out1, out2, out3, out4, out5, out6, out7, out8, out9] = outputs;
        let result = timed(&self.timings.native, || {
            self.native.submit_resident_batch10(
                handles,
                activations,
                [out0, out1, out2, out3, out4, out5, out6, out7, out8, out9],
            )
        });
        if result.is_ok() {
            self.failed = false;
        } else {
            for output in [out0, out1, out2, out3, out4, out5, out6, out7, out8, out9] {
                output.fill(f32::NAN);
            }
        }
        result
    }

    fn submit(
        &mut self,
        _: &[u8],
        _: &[[f32; 128]; 8],
        output: &mut [f32],
    ) -> std::result::Result<(), Error> {
        self.failed = true;
        output.fill(f32::NAN);
        Err(Error::Contract)
    }

    fn submit_batch4(
        &mut self,
        _: [&[u8]; 4],
        _: [&[[f32; 128]; 8]; 4],
        outputs: [&mut [f32]; 4],
    ) -> std::result::Result<(), Error> {
        self.failed = true;
        for output in outputs {
            output.fill(f32::NAN);
        }
        Err(Error::Contract)
    }

    fn submit_batch10(
        &mut self,
        _: [&[u8]; 10],
        _: [&[[f32; 128]; 8]; 10],
        outputs: [&mut [f32]; 10],
    ) -> std::result::Result<(), Error> {
        self.failed = true;
        for output in outputs {
            output.fill(f32::NAN);
        }
        Err(Error::Contract)
    }
}

struct TimedNativeQ2Submitter<'a> {
    native: &'a mut dyn NativeQ2Submitter,
    timings: &'a SubmitTimings,
    resident_only: bool,
    failed: bool,
}

impl NativeQ2Submitter for TimedNativeQ2Submitter<'_> {
    fn remaining_operations(&self) -> Option<u64> {
        if self.failed {
            Some(0)
        } else {
            self.native.remaining_operations()
        }
    }

    fn residency_capacity_bytes(&self) -> usize {
        if self.failed || !self.resident_only {
            0
        } else {
            self.native.residency_capacity_bytes()
        }
    }

    fn prepare_weights(
        &mut self,
        columns: usize,
        rows: usize,
        weights: &[u8],
    ) -> std::result::Result<Option<NativeQ2WeightHandle>, Error> {
        if self.failed || !self.resident_only {
            return Err(Error::Contract);
        }
        self.failed = true;
        let handle = self.native.prepare_weights(columns, rows, weights)?;
        if handle.is_some() {
            self.failed = false;
        }
        Ok(handle)
    }

    fn submit_resident(
        &mut self,
        handle: &NativeQ2WeightHandle,
        activation: &[f32],
        output: &mut [f32],
    ) -> std::result::Result<(), Error> {
        if self.failed || !self.resident_only {
            output.fill(f32::NAN);
            return Err(Error::Contract);
        }
        self.failed = true;
        let result = timed(&self.timings.native, || {
            self.native.submit_resident(handle, activation, output)
        });
        if result.is_ok() {
            self.failed = false;
        } else {
            output.fill(f32::NAN);
        }
        result
    }

    fn submit(
        &mut self,
        columns: usize,
        rows: usize,
        weights: &[u8],
        activation: &[f32],
        output: &mut [f32],
    ) -> std::result::Result<(), Error> {
        if self.failed || self.resident_only {
            output.fill(f32::NAN);
            return Err(Error::Contract);
        }
        self.failed = true;
        let result = timed(&self.timings.native, || {
            self.native
                .submit(columns, rows, weights, activation, output)
        });
        if result.is_ok() {
            self.failed = false;
        } else {
            output.fill(f32::NAN);
        }
        result
    }
}

#[cfg(any(windows, test))]
fn top5_indices(values: &[f32]) -> [usize; 5] {
    let mut top = [(0_usize, f32::NEG_INFINITY); 5];
    for (idx, &val) in values.iter().enumerate() {
        if val > top[4].1 {
            top[4] = (idx, val);
            for k in (0..4).rev() {
                if top[k + 1].1 > top[k].1 {
                    top.swap(k, k + 1);
                } else {
                    break;
                }
            }
        }
    }
    std::array::from_fn(|i| top[i].0)
}

#[cfg(any(windows, test))]
#[allow(
    clippy::as_conversions,
    clippy::cast_precision_loss,
    reason = "Vocabulary size (248,320) fits exactly in f64."
)]
fn compute_logits_tolerance(actual: &[f32], expected: &[f32]) -> Result<(bool, Value)> {
    require(actual.len() == expected.len() && !actual.is_empty())?;
    let mut max_abs_diff = 0.0_f64;
    let mut sum_abs_diff = 0.0_f64;
    let mut sum_sq_diff = 0.0_f64;
    let mut dot = 0.0_f64;
    let mut norm_actual = 0.0_f64;
    let mut norm_expected = 0.0_f64;
    for (&act, &exp) in actual.iter().zip(expected) {
        require(act.is_finite() && exp.is_finite())?;
        let act_f64 = f64::from(act);
        let exp_f64 = f64::from(exp);
        let diff = (act_f64 - exp_f64).abs();
        if diff > max_abs_diff {
            max_abs_diff = diff;
        }
        sum_abs_diff += diff;
        sum_sq_diff = f64::mul_add(diff, diff, sum_sq_diff);
        dot = f64::mul_add(act_f64, exp_f64, dot);
        norm_actual = f64::mul_add(act_f64, act_f64, norm_actual);
        norm_expected = f64::mul_add(exp_f64, exp_f64, norm_expected);
    }
    let count = actual.len() as f64;
    let mean_abs_diff = sum_abs_diff / count;
    let rmse = (sum_sq_diff / count).sqrt();
    let denom = (norm_actual * norm_expected).sqrt();
    let cosine_similarity = if denom > 0.0 { dot / denom } else { 1.0 };
    let cpu_top5 = top5_indices(expected);
    let npu_top5 = top5_indices(actual);
    let top1_match = cpu_top5[0] == npu_top5[0];
    let top5_overlap = npu_top5
        .iter()
        .filter(|&&idx| cpu_top5.contains(&idx))
        .count();
    let pass = cosine_similarity >= 0.995 && rmse <= 0.5 && max_abs_diff <= 5.0;
    Ok((
        pass,
        json!({
            "maxAbsDiff": max_abs_diff,
            "meanAbsDiff": mean_abs_diff,
            "rmse": rmse,
            "cosineSimilarity": cosine_similarity,
            "cpuTop1Token": cpu_top5[0],
            "npuTop1Token": npu_top5[0],
            "top1Match": top1_match,
            "top5Overlap": top5_overlap,
        }),
    ))
}

#[cfg(windows)]
fn compare_graph_sequence(
    graph: Qwen35,
    executor: &mut dyn Pq2ProjectionExecutor,
    placement_model: &Model,
    controls: &[Control],
    request: &Request,
    parent: &bonsai_windows_process::ParentProcess,
    timings: &SubmitTimings,
) -> Result<Value> {
    require(controls.len() == request.tokens().len())?;
    let projections = request.projection_names();
    let mut placed = PlacedProjectionExecutor::new(executor, placement_model, &projections)?;
    let mut projected = graph.with_projection_executor(&mut placed)?;
    // Reserve the complete sequence before the first submitted projection.
    projected.admit_work(0, request.tokens().len())?;
    let mut comparisons = Vec::new();
    comparisons.try_reserve_exact(controls.len())?;
    let mut all_logits_bitwise = true;
    let mut all_state_bitwise = true;
    for (index, (&token, control)) in request.tokens().iter().zip(controls).enumerate() {
        parent.ensure_alive()?;
        let submit_before = timings.native.get();
        let oracle_before = timings.oracle.get();
        let started = Instant::now();
        let actual = projected.evaluate(token)?;
        let evaluation_duration = started.elapsed();
        let token_timing = token_timings(
            control.evaluation_duration,
            evaluation_duration,
            timings.native.get().saturating_sub(submit_before),
            timings.oracle.get().saturating_sub(oracle_before),
        );
        let logits_bitwise = actual.len() == control.logits.len()
            && actual
                .iter()
                .zip(&control.logits)
                .all(|(actual, expected)| actual.to_bits() == expected.to_bits());
        let actual_state = state_digest(projected.decoder())?;
        let state_bitwise = compare_state(projected.decoder(), &control.rows).is_ok()
            && actual_state == control.state_sha256;
        all_logits_bitwise &= logits_bitwise;
        all_state_bitwise &= state_bitwise;
        if request.execution_backend == ExecutionBackend::NativeQ2 {
            let (tol_pass, tol_json) = compute_logits_tolerance(&actual, &control.logits)?;
            require(tol_pass)?;
            comparisons.push(json!({"index":index,"tokenId":token,
                "logitsBitwiseEqual":logits_bitwise,"stateBitwiseEqual":state_bitwise,
                "logitsBoundedTolerancePass":true,"numericalTolerance":tol_json,
                "logitsSha256":logits_digest(&actual)?,"stateSha256":actual_state,
                "timing":token_timing}));
        } else {
            require(logits_bitwise && state_bitwise)?;
            comparisons.push(json!({"index":index,"tokenId":token,
                "logitsBitwiseEqual":true,"stateBitwiseEqual":true,
                "logitsSha256":logits_digest(&actual)?,"stateSha256":actual_state,
                "timing":token_timing}));
        }
    }
    let last = comparisons.last().ok_or_else(reject)?;
    let mut summary = json!({"placementIdentity":projected.identity(),"tokenComparisons":comparisons,
        "logitsBitwiseEqual":all_logits_bitwise,"stateBitwiseEqual":all_state_bitwise,
        "logitsSha256":last["logitsSha256"],"stateSha256":last["stateSha256"]});
    if request.execution_backend == ExecutionBackend::NativeQ2 {
        summary["logitsBoundedTolerancePass"] = json!(true);
        summary["numericalTolerance"] = last["numericalTolerance"].clone();
    }
    Ok(summary)
}

fn check_statistics(
    statistics: bonsai_hexagon_runtime::ProjectionStatistics,
    compared: u64,
    failed: bool,
    request: &Request,
) -> Result<()> {
    let tokens = u64::try_from(request.tokens().len())?;
    let projections = u64::try_from(request.projection_names().len())?;
    let total_projections = tokens * projections;
    let ops_per_proj = u64::from(request.execution_backend.operations_per_projection());
    require(
        !failed
            && compared
                == if request.resident_host_weights {
                    0
                } else {
                    request.execution_backend.values_per_token() * total_projections
                }
            && statistics.projections_completed == total_projections
            && statistics.operations_completed == total_projections * ops_per_proj
            && statistics.operations_attempted == total_projections * ops_per_proj,
    )
}

#[cfg(windows)]
#[allow(
    clippy::too_many_lines,
    reason = "Keep the complete terminal CPU/native comparison and post-cleanup publication sequence auditable together."
)]
fn run_request(request: &Request, request_sha256: &str) -> Result<()> {
    let limits = request.validate()?;
    let projections = request.projection_names();
    let projection_count = projections.len();
    let parent = bonsai_windows_process::ParentProcess::open(
        request.expected_parent_pid,
        request.expected_parent_creation_time,
    )?;
    parent.ensure_alive()?;
    emit(
        &json!({"event":"started", "schemaVersion":1, "requestSha256":request_sha256,
        "processId":std::process::id(), "modelSha256":swift::MODEL_SHA256,
        "modelBytes":swift::MODEL_BYTES, "projection":PROJECTION, "projections":projections}),
    )?;
    let (model, _) = swift::load(&request.model_path, request.model_memory_limit_bytes)?;
    let mut selected_tensors = BTreeMap::new();
    for &name in &projections {
        let dims = projection_dimensions(name, request.execution_backend).ok_or_else(reject)?;
        let selected = model.tensors.get(name).ok_or_else(reject)?;
        require(selected.kind == 142 && selected.dimensions == <[usize; 2]>::from(dims))?;
        selected_tensors.insert(name.to_owned(), selected.clone());
    }
    let shadow = shared_model(&model);
    let placement_model = Model {
        metadata: BTreeMap::default(),
        tensors: selected_tensors,
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
    let graph = graph_for_arm(model, request, false)?;
    let batch_ops = request.batch_operations()?;
    let row_parallelism = RowParallelism::new(request.row_workers)?;
    parent.ensure_alive()?;
    let npu_power_before = power_value();
    let mut result = None;
    let mut progress_failed = false;
    let submit_timings = SubmitTimings::default();
    let report_result = (|| {
        let driver = Driver::open_system()?;
        let driver_dll = driver.path().to_path_buf();
        let progress = |report: &hexagon_native_rpc::Q2ProjectionReport| {
            if emit(
                &json!({"event":"native-progress", "stage":report.queue.stage,
                "batchOperations":report.batch_operations,
                "operationsSubmitted":report.operations_submitted,
                "operationsCompleted":report.operations_completed,
                "queueWritesSubmitted":report.queue_writes_submitted,
                "queueWritesCompleted":report.queue_writes_completed}),
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
                            timings: &submit_timings,
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
                                &submit_timings,
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
                let resident_weight_bytes = RESIDENT_WEIGHT_BYTES * projection_count;
                let mut hmx_limits = HmxQ2ProjectionLimits::new(
                    17_408,
                    request.maximum_operations,
                    Duration::from_secs(request.session_seconds),
                )?;
                if request.batch_ten_operations {
                    hmx_limits = hmx_limits.with_batch_ten()?;
                } else if request.batch_four_operations {
                    hmx_limits = hmx_limits.with_batch_four()?;
                }
                if request.resident_host_weights {
                    hmx_limits = hmx_limits.with_resident_weights(resident_weight_bytes)?;
                }
                driver.reference_hmx_q2_projection(
                    &request.package_directory,
                    hmx_limits,
                    progress,
                    |native| {
                        let work = (|| -> Result<()> {
                            let expected_writes =
                                u64::try_from(request.tokens().len() * projection_count)?
                                    * (40 / u64::from(batch_ops));
                            if request.resident_host_weights {
                                let mut timed = TimedHmxSubmitter {
                                    native,
                                    timings: &submit_timings,
                                    failed: false,
                                };
                                let (mut value, statistics, residency, writes_attempted, writes_completed) = {
                                    let mut executor = HmxQ2Executor::new(
                                        &mut timed,
                                        hmx_limits,
                                        request.maximum_owned_payload_bytes,
                                    )
                                    .with_row_parallelism(row_parallelism)?
                                    .with_resident_weights(RESIDENT_METADATA_BYTES)?;
                                    let value = compare_graph_sequence(
                                        graph,
                                        &mut executor,
                                        &placement_model,
                                        &controls,
                                        request,
                                        &parent,
                                        &submit_timings,
                                    )?;
                                    (
                                        value,
                                        executor.statistics(),
                                        executor.residency_statistics(),
                                        executor.queue_writes_attempted(),
                                        executor.queue_writes_completed(),
                                    )
                                };
                                check_statistics(statistics, 0, timed.failed, request)?;
                                require(
                                    writes_attempted == expected_writes
                                        && writes_completed == expected_writes,
                                )?;
                                let projections_u64 = u64::try_from(projection_count)?;
                                require(residency.tensors_prepared == projections_u64
                                    && residency.cache_hits == projections_u64 * u64::try_from(request.tokens().len() - 1)?
                                    && residency.weight_payload_bytes_prepared == u64::try_from(resident_weight_bytes)?
                                    && (1..=RESIDENT_METADATA_BYTES).contains(&residency.metadata_payload_bytes))?;
                                value["hostResidency"] = json!({
                                    "tensorsPrepared":residency.tensors_prepared,
                                    "cacheHits":residency.cache_hits,
                                    "weightPayloadBytesPrepared":residency.weight_payload_bytes_prepared,
                                    "metadataPayloadBytes":residency.metadata_payload_bytes,
                                });
                                return accept(value, statistics, 0);
                            }
                            let mut checked = CheckedHmxSubmitter {
                                native,
                                timings: &submit_timings,
                                compared: 0,
                                failed: false,
                            };
                            let (value, statistics, writes_attempted, writes_completed) = {
                                let mut executor = HmxQ2Executor::new(
                                    &mut checked,
                                    hmx_limits,
                                    request.maximum_owned_payload_bytes,
                                )
                                .with_row_parallelism(row_parallelism)?;
                                let value = compare_graph_sequence(
                                    graph,
                                    &mut executor,
                                    &placement_model,
                                    &controls,
                                    request,
                                    &parent,
                                    &submit_timings,
                                )?;
                                (
                                    value,
                                    executor.statistics(),
                                    executor.queue_writes_attempted(),
                                    executor.queue_writes_completed(),
                                )
                            };
                            check_statistics(
                                statistics,
                                checked.compared,
                                checked.failed,
                                request,
                            )?;
                            require(
                                writes_attempted == expected_writes
                                    && writes_completed == expected_writes,
                            )?;
                            accept(value, statistics, checked.compared)
                        })();
                        work.map_err(|_| Error::Contract)
                    },
                )
            }
            ExecutionBackend::NativeQ2 => {
                let resident_weight_bytes =
                    expected_resident_weight_bytes(&projections, request.execution_backend)
                        .map_err(|_| Error::Contract)?;
                let max_columns = projections
                    .iter()
                    .filter_map(|&name| {
                        projection_dimensions(name, request.execution_backend).map(|(c, _)| c)
                    })
                    .max()
                    .unwrap_or(5120);
                let max_rows = projections
                    .iter()
                    .filter_map(|&name| {
                        projection_dimensions(name, request.execution_backend).map(|(_, r)| r)
                    })
                    .max()
                    .unwrap_or(17_408);
                let mut native_limits = NativeQ2ProjectionLimits::new(
                    max_columns,
                    max_rows,
                    request.maximum_operations,
                    Duration::from_secs(request.session_seconds),
                )?;
                if request.resident_host_weights {
                    native_limits = native_limits.with_resident_weights(resident_weight_bytes)?;
                }
                driver.reference_native_q2_projection(
                    &request.package_directory,
                    native_limits,
                    progress,
                    |native| {
                        let work = (|| -> Result<()> {
                            let expected_writes =
                                u64::try_from(request.tokens().len() * projection_count)?;
                            let mut timed = TimedNativeQ2Submitter {
                                native,
                                timings: &submit_timings,
                                resident_only: request.resident_host_weights,
                                failed: false,
                            };
                            let (
                                mut value,
                                statistics,
                                residency,
                                writes_attempted,
                                writes_completed,
                            ) = {
                                let mut executor = NativeQ2Executor::new(
                                    &mut timed,
                                    native_limits,
                                    request.maximum_owned_payload_bytes,
                                )
                                .with_row_parallelism(row_parallelism)?;
                                if request.resident_host_weights {
                                    executor = executor.with_resident_weights(
                                        NATIVE_Q2_RESIDENT_METADATA_BYTES,
                                    )?;
                                    for &name in &projections {
                                        let tensor = placement_model
                                            .tensors
                                            .get(name)
                                            .ok_or_else(reject)?;
                                        require(executor.prewarm_tensor(tensor)?)?;
                                    }
                                }
                                let value = compare_graph_sequence(
                                    graph,
                                    &mut executor,
                                    &placement_model,
                                    &controls,
                                    request,
                                    &parent,
                                    &submit_timings,
                                )?;
                                (
                                    value,
                                    executor.statistics(),
                                    executor.residency_statistics(),
                                    executor.queue_writes_attempted(),
                                    executor.queue_writes_completed(),
                                )
                            };
                            check_statistics(statistics, 0, timed.failed, request)?;
                            require(
                                writes_attempted == expected_writes
                                    && writes_completed == expected_writes,
                            )?;
                            if request.resident_host_weights {
                                let projections_u64 = u64::try_from(projection_count)?;
                                require(
                                    residency.tensors_prepared == projections_u64
                                        && residency.cache_hits
                                            == projections_u64
                                                * u64::try_from(request.tokens().len() - 1)?
                                        && residency.weight_payload_bytes_prepared
                                            == u64::try_from(resident_weight_bytes)?
                                        && (1..=NATIVE_Q2_RESIDENT_METADATA_BYTES)
                                            .contains(&residency.metadata_payload_bytes),
                                )?;
                                value["hostResidency"] = json!({
                                    "tensorsPrepared":residency.tensors_prepared,
                                    "cacheHits":residency.cache_hits,
                                    "weightPayloadBytesPrepared":residency.weight_payload_bytes_prepared,
                                    "metadataPayloadBytes":residency.metadata_payload_bytes,
                                });
                            }
                            accept(value, statistics, 0)
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
    let expected_batch = request.batch_operations()?;
    let ops_per_proj = request.execution_backend.operations_per_projection();
    let expected_writes =
        u32::try_from(request.tokens().len() * projection_count)? * (ops_per_proj / expected_batch);
    require(
        report.batch_operations == expected_batch
            && report.queue_writes_submitted == expected_writes
            && report.queue_writes_completed == expected_writes,
    )?;
    let mut value = result.ok_or_else(reject)?;
    value["event"] = json!("result");
    value["schemaVersion"] = json!(1);
    value["completed"] = json!(true);
    value["requestSha256"] = json!(request_sha256);
    value["modelSha256"] = json!(swift::MODEL_SHA256);
    value["modelBytes"] = json!(swift::MODEL_BYTES);
    value["projection"] = json!(PROJECTION);
    value["projections"] = json!(projections);
    value["tokenId"] = json!(request.token_id);
    value["tokenIds"] = json!(request.tokens());
    value["executionBackend"] = json!(request.execution_backend.name());
    value["residentHostWeights"] = json!(request.resident_host_weights);
    value["batchFourOperations"] = json!(request.batch_four_operations);
    value["batchTenOperations"] = json!(request.batch_ten_operations);
    value["integerAudit"] = json!(if request.execution_backend == ExecutionBackend::NativeQ2 {
        "bounded-float-tolerance"
    } else if request.resident_host_weights {
        "graph-parity-only"
    } else {
        "all-returned-integers"
    });
    value["adapterCase"] = json!(request.adapter_case.name());
    value["adapterLoaded"] = json!(request.adapter_case.loaded());
    value["adapterSha256"] = if request.adapter_case.loaded() {
        json!(bonsai_prism_runtime::model::lora::SWIFT_ADAPTER_SHA256)
    } else {
        Value::Null
    };
    value["adapterScaleBits"] = json!(request.adapter_case.scale().to_bits());
    value["cpuBaselineAdapterLoaded"] = json!(request.adapter_case == AdapterCase::StandardScale13);
    value["cpuBaselineAdapterScaleBits"] = json!(request.adapter_case.scale().to_bits());
    value["cpuBackend"] = json!(request.cpu_backend.name());
    value["rowWorkers"] = json!(request.row_workers);
    value["requiredControlStateBytes"] = json!(control_state_bytes(request.tokens().len())?);
    value["maximumOperations"] = json!(request.maximum_operations);
    value["sessionSeconds"] = json!(request.session_seconds);
    value["nativeHmxBitplaneControl"] = json!(report.is_hmx_bitplane_control());
    value["profileEligible"] = json!(false);
    value["performanceMeasured"] = json!(false);
    value["timingScope"] = json!({
        "kind":"single-run-diagnostic-evaluation-windows",
        "includes":"All CPU-owned graph work, offload packing/reconstruction, native submission and wait inside evaluate; checked native evaluation also includes the independently timed integer oracle.",
        "excludes":"Model loading, graph construction, device setup/cleanup, state snapshots/digests and final graph comparisons.",
        "qualifiedBenefitComparison":false,
        "subtractedOracleWindowIsEstimated":!request.resident_host_weights,
    });
    value["cpuPowerBefore"] = cpu_power_before;
    value["cpuPowerAfter"] = cpu_power_after;
    value["npuPowerBefore"] = npu_power_before;
    value["nativeCompletedCycle"] = json!(true);
    value["nativeBatchOperations"] = json!(report.batch_operations);
    value["nativeOperationsSubmitted"] = json!(report.operations_submitted);
    value["nativeOperationsCompleted"] = json!(report.operations_completed);
    value["nativeQueueWritesSubmitted"] = json!(report.queue_writes_submitted);
    value["nativeQueueWritesCompleted"] = json!(report.queue_writes_completed);
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
        |error| {
            let mut failure = json!({"event":"error","schemaVersion":1,"completed":false,
                "reason":"hexagon-model-probe-rejected","profileEligible":false});
            if let Some(budget) = error.downcast_ref::<ControlStateBudgetExceeded>() {
                failure["reason"] = json!("control-snapshot-budget");
                failure["requiredControlStateBytes"] = json!(budget.required);
                failure["stateBudgetBytes"] = json!(budget.available);
            }
            let _ = emit(&failure);
            ExitCode::FAILURE
        },
        |()| ExitCode::SUCCESS,
    )
}

#[cfg(test)]
mod tests {
    use std::rc::Rc;

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
            ("tokenIds", json!([0, 1, 2, 3, 4, 5, 6, 7, 8])),
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
        let mut eight = request();
        eight["tokenIds"] = json!([0, 1, 2, 3, 4, 5, 6, 7]);
        eight["contextLimit"] = json!(8);
        eight["maximumOperations"] = json!(320);
        eight["stateBudgetBytes"] = json!(control_state_bytes(8)?);
        serde_json::from_value::<Request>(eight.clone())?.limits()?;
        eight["maximumOperations"] = json!(319);
        assert!(serde_json::from_value::<Request>(eight)?.limits().is_err());
        Ok(())
    }

    #[test]
    fn model_probe_admits_complete_cpu_snapshot_retention_before_model_work() -> Result<()> {
        assert_eq!(
            CONTROL_RECURRENT_BYTES + CONTROL_KV_BYTES_PER_TOKEN / 2 * swift::CONTEXT_TOKENS,
            swift::STATE_PAYLOAD_BYTES
        );
        assert_eq!(control_state_bytes(3)?, 471_465_984);
        assert_eq!(control_state_bytes(4)?, 628_883_456);
        assert!(control_state_bytes(0).is_err());
        assert!(control_state_bytes(MAX_TOKENS + 1).is_err());
        let mut three = request();
        three["tokenIds"] = json!([0, 42, 43]);
        three["contextLimit"] = json!(3);
        serde_json::from_value::<Request>(three)?.limits()?;
        let mut four = request();
        four["tokenIds"] = json!([0, 42, 43, 44]);
        four["contextLimit"] = json!(4);
        let error = serde_json::from_value::<Request>(four.clone())?
            .limits()
            .err()
            .ok_or_else(reject)?;
        let budget = error
            .downcast_ref::<ControlStateBudgetExceeded>()
            .ok_or_else(reject)?;
        assert_eq!(budget.required, 628_883_456);
        assert_eq!(budget.available, 536_870_912);
        four["stateBudgetBytes"] = json!(628_883_456);
        serde_json::from_value::<Request>(four.clone())?.limits()?;
        four["stateBudgetBytes"] = json!(628_883_455);
        assert!(serde_json::from_value::<Request>(four)?.limits().is_err());
        Ok(())
    }

    #[test]
    fn model_probe_cpu_failure_reports_stage_and_index_without_a_success_claim() -> Result<()> {
        let mut events = Vec::new();
        let result: Result<()> = cpu_stage_with_report(
            "state-capture",
            Some(3),
            || Err(reject()),
            |value| {
                events.push(value.clone());
                Ok(())
            },
        );
        assert!(result.is_err());
        assert_eq!(
            events,
            vec![json!({"event":"cpu-control-failure","stage":"state-capture","tokenIndex":3})]
        );
        events.clear();
        cpu_stage_with_report(
            "evaluate",
            Some(0),
            || Ok(()),
            |value| {
                events.push(value.clone());
                Ok(())
            },
        )?;
        assert_eq!(events, [Value::Null; 0]);
        let result: Result<()> = cpu_stage_with_report(
            "graph-construction",
            None,
            || Err(reject()),
            |value| {
                events.push(value.clone());
                Ok(())
            },
        );
        assert!(result.is_err());
        assert_eq!(events[0]["stage"], "graph-construction");
        assert!(events[0]["tokenIndex"].is_null());
        Ok(())
    }

    #[test]
    fn model_probe_adapter_case_requires_an_explicit_path_and_approved_scale() -> Result<()> {
        let value = request();
        for case in ["loaded-zero", "standard-scale-1.3"] {
            let mut changed = value.clone();
            changed["adapterCase"] = json!(case);
            assert!(
                serde_json::from_value::<Request>(changed.clone())?
                    .limits()
                    .is_err()
            );
            changed["adapterPath"] = json!("C:\\approved-adapter.gguf");
            let parsed: Request = serde_json::from_value(changed)?;
            parsed.limits()?;
            assert!(parsed.adapter_case.loaded());
            assert_eq!(parsed.adapter_case.name(), case);
            assert_eq!(
                parsed.adapter_case.scale().to_bits(),
                if case == "loaded-zero" {
                    0
                } else {
                    1.3_f32.to_bits()
                }
            );
        }
        let mut extra = value.clone();
        extra["adapterPath"] = json!("C:\\approved-adapter.gguf");
        assert!(serde_json::from_value::<Request>(extra)?.limits().is_err());
        for invalid_case in ["standard", "standard-scale-1.0", "arbitrary"] {
            let mut invalid = value.clone();
            invalid["adapterCase"] = json!(invalid_case);
            assert!(serde_json::from_value::<Request>(invalid).is_err());
        }
        Ok(())
    }

    #[test]
    fn model_probe_cpu_policy_rejects_silent_worker_and_neon_fallback() -> Result<()> {
        for workers in [0, 2, 3, 5, 16] {
            let mut invalid = request();
            invalid["rowWorkers"] = json!(workers);
            assert!(
                serde_json::from_value::<Request>(invalid)?
                    .limits()
                    .is_err()
            );
        }
        let mut four = request();
        four["rowWorkers"] = json!(4);
        assert_eq!(
            serde_json::from_value::<Request>(four)?.limits().is_ok(),
            cfg!(feature = "parallel-rows")
        );
        let mut neon = request();
        neon["cpuBackend"] = json!("neon");
        assert_eq!(
            serde_json::from_value::<Request>(neon)?.limits().is_ok(),
            CpuBackend::Neon.available()
        );
        let mut automatic = request();
        automatic["cpuBackend"] = json!("auto");
        assert!(serde_json::from_value::<Request>(automatic).is_err());
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

        fn submit_resident(
            &mut self,
            _: &HmxQ2WeightHandle,
            activation: &[[f32; 128]; 8],
            output: &mut [f32],
        ) -> std::result::Result<(), Error> {
            self.submit(&[], activation, output)
        }

        fn submit_batch4(
            &mut self,
            _: [&[u8]; 4],
            _: [&[[f32; 128]; 8]; 4],
            outputs: [&mut [f32]; 4],
        ) -> std::result::Result<(), Error> {
            self.calls.set(self.calls.get() + 1);
            for output in outputs {
                output.fill(if self.wrong { 1.0 } else { 0.0 });
            }
            if self.fail {
                Err(Error::Contract)
            } else {
                Ok(())
            }
        }

        fn submit_resident_batch4(
            &mut self,
            _: [&HmxQ2WeightHandle; 4],
            activations: [&[[f32; 128]; 8]; 4],
            outputs: [&mut [f32]; 4],
        ) -> std::result::Result<(), Error> {
            self.submit_batch4([&[], &[], &[], &[]], activations, outputs)
        }

        fn submit_batch10(
            &mut self,
            _: [&[u8]; 10],
            _: [&[[f32; 128]; 8]; 10],
            outputs: [&mut [f32]; 10],
        ) -> std::result::Result<(), Error> {
            self.calls.set(self.calls.get() + 1);
            for output in outputs {
                output.fill(if self.wrong { 1.0 } else { 0.0 });
            }
            if self.fail {
                Err(Error::Contract)
            } else {
                Ok(())
            }
        }

        fn submit_resident_batch10(
            &mut self,
            _: [&HmxQ2WeightHandle; 10],
            activations: [&[[f32; 128]; 8]; 10],
            outputs: [&mut [f32]; 10],
        ) -> std::result::Result<(), Error> {
            self.submit_batch10(
                [&[], &[], &[], &[], &[], &[], &[], &[], &[], &[]],
                activations,
                outputs,
            )
        }
    }

    #[test]
    fn model_probe_batch4_requires_hmx_backend_and_checks_all_four_members() -> Result<()> {
        let mut value = request();
        value["batchFourOperations"] = json!(true);
        assert!(
            serde_json::from_value::<Request>(value.clone())?
                .limits()
                .is_err()
        );
        value["executionBackend"] = json!("hmx-bitplane");
        serde_json::from_value::<Request>(value.clone())?.limits()?;
        value["maximumOperations"] = json!(42);
        assert!(serde_json::from_value::<Request>(value)?.limits().is_err());

        let calls = Rc::new(Cell::new(0));
        let mut native = HmxSubmitter {
            calls: calls.clone(),
            fail: false,
            wrong: false,
        };
        let timings = SubmitTimings::default();
        let mut checked = CheckedHmxSubmitter {
            native: &mut native,
            timings: &timings,
            compared: 0,
            failed: false,
        };
        let weights = zero_hmx_weights(32);
        let activation = [[1.0; 128]; 8];
        let mut out0 = [f32::NAN; 256];
        let mut out1 = [f32::NAN; 256];
        let mut out2 = [f32::NAN; 256];
        let mut out3 = [f32::NAN; 256];
        checked.submit_batch4(
            [&weights, &weights, &weights, &weights],
            [&activation, &activation, &activation, &activation],
            [&mut out0, &mut out1, &mut out2, &mut out3],
        )?;
        assert_eq!(calls.get(), 1);
        assert_eq!(checked.compared, 1024);
        assert!(!checked.failed);
        Ok(())
    }

    #[test]
    fn model_probe_batch10_requires_hmx_backend_and_checks_all_ten_members() -> Result<()> {
        let mut value = request();
        value["batchTenOperations"] = json!(true);
        assert!(
            serde_json::from_value::<Request>(value.clone())?
                .limits()
                .is_err()
        );
        value["executionBackend"] = json!("hmx-bitplane");
        serde_json::from_value::<Request>(value.clone())?.limits()?;
        let mut conflicting = value.clone();
        conflicting["batchFourOperations"] = json!(true);
        assert!(
            serde_json::from_value::<Request>(conflicting)?
                .limits()
                .is_err()
        );
        value["maximumOperations"] = json!(45);
        assert!(serde_json::from_value::<Request>(value)?.limits().is_err());

        let calls = Rc::new(Cell::new(0));
        let mut native = HmxSubmitter {
            calls: calls.clone(),
            fail: false,
            wrong: false,
        };
        let timings = SubmitTimings::default();
        let mut checked = CheckedHmxSubmitter {
            native: &mut native,
            timings: &timings,
            compared: 0,
            failed: false,
        };
        let weights = zero_hmx_weights(32);
        let activation = [[1.0; 128]; 8];
        let mut out0 = [f32::NAN; 256];
        let mut out1 = [f32::NAN; 256];
        let mut out2 = [f32::NAN; 256];
        let mut out3 = [f32::NAN; 256];
        let mut out4 = [f32::NAN; 256];
        let mut out5 = [f32::NAN; 256];
        let mut out6 = [f32::NAN; 256];
        let mut out7 = [f32::NAN; 256];
        let mut out8 = [f32::NAN; 256];
        let mut out9 = [f32::NAN; 256];
        checked.submit_batch10(
            [&weights; 10],
            [&activation; 10],
            [
                &mut out0, &mut out1, &mut out2, &mut out3, &mut out4, &mut out5, &mut out6,
                &mut out7, &mut out8, &mut out9,
            ],
        )?;
        assert_eq!(calls.get(), 1);
        assert_eq!(checked.compared, 2560);
        assert!(!checked.failed);
        Ok(())
    }

    #[test]
    fn model_probe_resident_timing_mode_binds_audit_and_refuses_streaming() -> Result<()> {
        let mut value = request();
        value["residentHostWeights"] = json!(true);
        assert!(
            serde_json::from_value::<Request>(value.clone())?
                .limits()
                .is_err()
        );
        value["executionBackend"] = json!("hmx-bitplane");
        let resident: Request = serde_json::from_value(value)?;
        resident.limits()?;
        let statistics = bonsai_hexagon_runtime::ProjectionStatistics {
            projections_completed: 1,
            operations_attempted: 40,
            operations_completed: 40,
        };
        check_statistics(statistics, 0, false, &resident)?;
        assert!(check_statistics(statistics, 1, false, &resident).is_err());
        assert!(check_statistics(statistics, 0, true, &resident).is_err());
        let mut registry = hexagon_native_rpc::HmxQ2WeightRegistry::new(4096)?;
        let handle = registry
            .prepare(&zero_hmx_weights(32))?
            .ok_or_else(reject)?;
        let calls = Rc::new(Cell::new(0));
        let mut native = HmxSubmitter {
            calls: calls.clone(),
            fail: false,
            wrong: false,
        };
        let timings = SubmitTimings::default();
        let mut timed = TimedHmxSubmitter {
            native: &mut native,
            timings: &timings,
            failed: false,
        };
        let mut output = [f32::NAN; 256];
        timed.submit_resident(&handle, &[[1.0; 128]; 8], &mut output)?;
        assert_eq!(calls.get(), 1);
        assert_eq!(timings.oracle.get(), Duration::ZERO);
        assert!(!timed.failed);
        assert!(
            timed
                .submit(&zero_hmx_weights(32), &[[1.0; 128]; 8], &mut output)
                .is_err()
        );
        assert!(output.iter().all(|value| value.is_nan()));
        assert!(
            timed
                .submit_resident(&handle, &[[1.0; 128]; 8], &mut output)
                .is_err()
        );
        assert_eq!(calls.get(), 1);
        assert_eq!(timed.remaining_operations(), Some(0));
        Ok(())
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
            let timings = SubmitTimings::default();
            let mut checked = CheckedHmxSubmitter {
                native: &mut native,
                timings: &timings,
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
        let timings = SubmitTimings::default();
        let mut checked = CheckedHmxSubmitter {
            native: &mut native,
            timings: &timings,
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
            let timings = SubmitTimings::default();
            let mut checked = CheckedSubmitter {
                native: &mut native,
                timings: &timings,
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
        let timings = SubmitTimings::default();
        let mut checked = CheckedSubmitter {
            native: &mut native,
            timings: &timings,
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

    #[test]
    fn model_probe_timing_windows_keep_native_wait_and_oracle_separate() {
        let timing = token_timings(
            Duration::from_millis(100),
            Duration::from_millis(180),
            Duration::from_millis(30),
            Duration::from_millis(50),
        );
        assert_eq!(timing["cpuEvaluationMilliseconds"], json!(100.0));
        assert_eq!(timing["nativeCheckedEvaluationMilliseconds"], json!(180.0));
        assert_eq!(timing["nativeSubmitAndWaitMilliseconds"], json!(30.0));
        assert_eq!(timing["independentIntegerOracleMilliseconds"], json!(50.0));
        assert_eq!(
            timing["nativeEvaluationExcludingIntegerOracleMilliseconds"],
            json!(130.0)
        );
        let accumulated = Cell::new(Duration::from_secs(1));
        let failure: Result<()> = timed(&accumulated, || Err(reject()));
        assert!(failure.is_err());
        assert!(accumulated.get() >= Duration::from_secs(1));
    }

    #[test]
    fn model_probe_multi_projection_admission_bounds_resident_arena_and_operations() -> Result<()> {
        let mut two_proj = request();
        two_proj["tokenIds"] = json!([42, 43, 128]);
        two_proj["tokenId"] = json!(42);
        two_proj["contextLimit"] = json!(3);
        two_proj["executionBackend"] = json!("hmx-bitplane");
        two_proj["residentHostWeights"] = json!(true);
        two_proj["batchFourOperations"] = json!(true);
        two_proj["projections"] = json!(["blk.0.ffn_up.weight", "blk.0.ffn_gate.weight"]);
        two_proj["maximumOperations"] = json!(240);
        let parsed: Request = serde_json::from_value(two_proj.clone())?;
        parsed.limits()?;
        assert_eq!(
            parsed.projection_names(),
            vec!["blk.0.ffn_up.weight", "blk.0.ffn_gate.weight"]
        );
        check_statistics(
            bonsai_hexagon_runtime::ProjectionStatistics {
                projections_completed: 6,
                operations_attempted: 240,
                operations_completed: 240,
            },
            0,
            false,
            &parsed,
        )?;
        assert!(
            check_statistics(
                bonsai_hexagon_runtime::ProjectionStatistics {
                    projections_completed: 3,
                    operations_attempted: 240,
                    operations_completed: 240,
                },
                0,
                false,
                &parsed,
            )
            .is_err()
        );
        for (field, invalid) in [
            ("projections", json!([])),
            ("projections", json!(["blk.0.ffn_gate.weight"])),
            (
                "projections",
                json!(["blk.0.ffn_up.weight", "blk.0.ffn_up.weight"]),
            ),
            (
                "projections",
                json!(["blk.0.ffn_up.weight", "blk.0.ffn_down.weight"]),
            ),
            (
                "projections",
                json!([
                    "blk.0.ffn_up.weight",
                    "blk.0.ffn_gate.weight",
                    "blk.1.ffn_up.weight"
                ]),
            ),
            ("maximumOperations", json!(236)),
        ] {
            let mut changed = two_proj.clone();
            changed[field] = invalid;
            assert!(
                serde_json::from_value::<Request>(changed)?
                    .limits()
                    .is_err()
            );
        }
        Ok(())
    }

    #[test]
    #[allow(
        clippy::too_many_lines,
        reason = "Keep native-q2 admission, multi-layer projection geometry and bounded tolerance checks together."
    )]
    fn model_probe_native_q2_admission_and_bounded_tolerance() -> Result<()> {
        let mut native_req = request();
        native_req["tokenIds"] = json!([42, 43, 128]);
        native_req["tokenId"] = json!(42);
        native_req["contextLimit"] = json!(3);
        native_req["executionBackend"] = json!("native-q2");
        native_req["residentHostWeights"] = json!(true);
        native_req["projections"] = json!(["blk.0.ffn_up.weight", "blk.0.ffn_gate.weight"]);
        native_req["maximumOperations"] = json!(6);
        let parsed: Request = serde_json::from_value(native_req.clone())?;
        assert!(parsed.execution_backend == ExecutionBackend::NativeQ2);
        assert_eq!(parsed.execution_backend.name(), "native-q2");
        assert_eq!(parsed.execution_backend.operations_per_projection(), 1);
        assert_eq!(parsed.execution_backend.values_per_token(), 0);
        parsed.limits()?;
        check_statistics(
            bonsai_hexagon_runtime::ProjectionStatistics {
                projections_completed: 6,
                operations_attempted: 6,
                operations_completed: 6,
            },
            0,
            false,
            &parsed,
        )?;
        let mut wrong_ops = native_req.clone();
        wrong_ops["maximumOperations"] = json!(5);
        assert!(
            serde_json::from_value::<Request>(wrong_ops)?
                .limits()
                .is_err()
        );
        let mut wrong_batch = native_req.clone();
        wrong_batch["batchFourOperations"] = json!(true);
        assert!(
            serde_json::from_value::<Request>(wrong_batch)?
                .limits()
                .is_err()
        );

        let full_layers = [
            "blk.0.ffn_up.weight",
            "blk.0.ffn_gate.weight",
            "blk.0.ffn_down.weight",
            "blk.0.attn_qkv.weight",
            "blk.0.attn_gate.weight",
            "blk.0.ssm_out.weight",
            "blk.3.ffn_up.weight",
            "blk.3.ffn_gate.weight",
            "blk.3.ffn_down.weight",
            "blk.3.attn_q.weight",
            "blk.3.attn_k.weight",
            "blk.3.attn_v.weight",
            "blk.3.attn_output.weight",
        ];
        assert_eq!(
            expected_resident_weight_bytes(&full_layers, ExecutionBackend::NativeQ2)?,
            212_336_640
        );
        let mut multi_layer_req = native_req.clone();
        multi_layer_req["projections"] = json!(full_layers);
        multi_layer_req["maximumOperations"] = json!(39);
        serde_json::from_value::<Request>(multi_layer_req)?.limits()?;

        let four_layer_block = [
            "blk.0.ffn_up.weight",
            "blk.0.ffn_gate.weight",
            "blk.0.ffn_down.weight",
            "blk.0.attn_qkv.weight",
            "blk.0.attn_gate.weight",
            "blk.0.ssm_out.weight",
            "blk.1.ffn_up.weight",
            "blk.1.ffn_gate.weight",
            "blk.1.ffn_down.weight",
            "blk.1.attn_qkv.weight",
            "blk.1.attn_gate.weight",
            "blk.1.ssm_out.weight",
            "blk.2.ffn_up.weight",
            "blk.2.ffn_gate.weight",
            "blk.2.ffn_down.weight",
            "blk.2.attn_qkv.weight",
            "blk.2.attn_gate.weight",
            "blk.2.ssm_out.weight",
            "blk.3.ffn_up.weight",
            "blk.3.ffn_gate.weight",
            "blk.3.ffn_down.weight",
            "blk.3.attn_q.weight",
            "blk.3.attn_k.weight",
            "blk.3.attn_v.weight",
            "blk.3.attn_output.weight",
        ];
        assert_eq!(
            expected_resident_weight_bytes(&four_layer_block, ExecutionBackend::NativeQ2)?,
            427_622_400
        );
        let mut four_layer_req = native_req.clone();
        four_layer_req["projections"] = json!(four_layer_block);
        four_layer_req["maximumOperations"] = json!(75);
        serde_json::from_value::<Request>(four_layer_req)?.limits()?;

        let default_proj = bonsai_hexagon_runtime::DEFAULT_NATIVE_Q2_PROJECTIONS;
        assert_eq!(default_proj.len(), 168);
        assert_eq!(
            expected_resident_weight_bytes(&default_proj[..28], ExecutionBackend::NativeQ2)?,
            502_824_960
        );
        assert_eq!(
            expected_resident_weight_bytes(&default_proj[..56], ExecutionBackend::NativeQ2)?,
            962_887_680
        );
        assert_eq!(
            expected_resident_weight_bytes(&default_proj[..112], ExecutionBackend::NativeQ2)?,
            1_925_775_360
        );
        assert_eq!(
            expected_resident_weight_bytes(&default_proj, ExecutionBackend::NativeQ2)?,
            2_888_663_040
        );
        let mut single_arena_req = native_req.clone();
        single_arena_req["projections"] = json!(&default_proj[..28]);
        single_arena_req["maximumOperations"] = json!(84);
        serde_json::from_value::<Request>(single_arena_req)?.limits()?;

        let mut dual_arena_req = native_req.clone();
        dual_arena_req["projections"] = json!(&default_proj[..56]);
        dual_arena_req["maximumOperations"] = json!(168);
        serde_json::from_value::<Request>(dual_arena_req)?.limits()?;

        let mut quad_arena_req = native_req.clone();
        quad_arena_req["projections"] = json!(&default_proj[..112]);
        quad_arena_req["maximumOperations"] = json!(336);
        serde_json::from_value::<Request>(quad_arena_req)?.limits()?;

        let mut six_arena_req = native_req;
        six_arena_req["projections"] = json!(&default_proj[..]);
        six_arena_req["maximumOperations"] = json!(504);
        serde_json::from_value::<Request>(six_arena_req)?.limits()?;

        let expected = [1.0_f32, 2.0, 3.0, 10.0, 5.0, 4.0];
        let actual = [1.01_f32, 1.99, 3.02, 9.98, 5.01, 3.99];
        let (pass, tol) = compute_logits_tolerance(&actual, &expected)?;
        assert!(pass);
        assert_eq!(tol["cpuTop1Token"], json!(3));
        assert_eq!(tol["npuTop1Token"], json!(3));
        assert_eq!(tol["top1Match"], json!(true));
        assert_eq!(tol["top5Overlap"], json!(5));
        let diverged = [-10.0_f32, -2.0, -3.0, -1.0, -5.0, -4.0];
        let (div_pass, _) = compute_logits_tolerance(&diverged, &expected)?;
        assert!(!div_pass);
        Ok(())
    }
}
