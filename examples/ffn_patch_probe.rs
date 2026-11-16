#![forbid(unsafe_code)]
//! Bounded explicit teacher-forced FFN capture; never a serving or quality gate.

use std::collections::BTreeSet;
use std::fs::{self, File, OpenOptions};
use std::io::{Read as _, Write as _};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use bonsai_native_hash::Hasher;
use bonsai_prism_runtime::graph::weight_patch::{
    EXECUTION_PROFILE, FfnPatch, FfnProbe, GROUP_OMISSION_PROFILE, INPUT_BASIS,
    SURROGATE_INPUT_BASIS, SURROGATE_TARGET_BASIS,
};
use bonsai_prism_runtime::graph::{MAX_STATE_BYTES, Qwen35};
use bonsai_prism_runtime::kernels::MatvecBackend;
use bonsai_prism_runtime::model::lora::{RankOneAdapter, SWIFT_ADAPTER_SHA256};
use bonsai_prism_runtime::model::swift::{self, Representation};
use bonsai_prism_runtime::{Decoder, Result};
use serde::Deserialize;
use serde_json::{Value, json};

const MAX_REQUEST_BYTES: u64 = 256 * 1024;
const MAX_PATCH_BYTES: u64 = 4 * 1024 * 1024;
const MAX_CAPTURE_BYTES: usize = 32 * 1024 * 1024;
const MAX_CAPTURE_ROWS: usize = 64;
const MAX_TOKENS: usize = 256;
const MODEL_BUDGET: usize = 8 * 1024 * 1024 * 1024;

#[path = "ffn_patch_probe/group_compensation.rs"]
mod group_compensation;
#[path = "ffn_patch_probe/group_omission.rs"]
mod group_omission;

#[derive(Clone, Copy, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
enum CaptureKind {
    #[default]
    HiddenPatch,
    FfnSurrogate,
    FfnGroupOmission,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Request {
    schema_version: u32,
    #[serde(default)]
    capture_kind: CaptureKind,
    model_path: PathBuf,
    adapter_path: Option<PathBuf>,
    adapter_scale_bits: String,
    weight_representation: String,
    layer: usize,
    runtime_source_commit: String,
    absolute_deadline_unix_ms: u64,
    capture_path: PathBuf,
    patch_path: Option<PathBuf>,
    #[serde(default)]
    omitted_groups: Option<Vec<usize>>,
    sequences: Vec<Sequence>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Sequence {
    sample_id: String,
    group_id: String,
    split: String,
    #[serde(default)]
    label: Option<String>,
    token_ids: Vec<u32>,
    capture_token_indices: Vec<usize>,
}

struct Deadline(Instant);

impl Deadline {
    fn check(&self) -> Result<()> {
        if Instant::now() >= self.0 {
            return Err(reject("FFN probe original deadline expired"));
        }
        Ok(())
    }
}

fn reject(message: &str) -> bonsai_prism_runtime::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidInput, message).into()
}

fn main() -> ExitCode {
    match run() {
        Ok(record) => {
            let mut output = std::io::stdout().lock();
            if serde_json::to_writer(&mut output, &record).is_err()
                || output.write_all(b"\n").is_err()
            {
                return ExitCode::FAILURE;
            }
            ExitCode::SUCCESS
        }
        Err(error) => {
            let _ = serde_json::to_writer(
                std::io::stderr().lock(),
                &json!({"event":"ffn-probe-failed","error":error.to_string()}),
            );
            ExitCode::FAILURE
        }
    }
}

fn read_bounded(path: &Path, limit: u64) -> Result<Vec<u8>> {
    if !path.is_absolute() {
        return Err(reject("FFN probe requires absolute input paths"));
    }
    let mut bytes = Vec::new();
    File::open(path)?.take(limit + 1).read_to_end(&mut bytes)?;
    if u64::try_from(bytes.len())? > limit {
        return Err(reject("FFN probe JSON exceeds its byte bound"));
    }
    Ok(bytes)
}

fn hash_file(path: &Path, deadline: &Deadline) -> Result<String> {
    let mut file = File::open(path)?;
    let mut hash = Hasher::new()?;
    let mut buffer = [0_u8; 8 * 1024];
    loop {
        deadline.check()?;
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count])?;
    }
    hex(hash.finish()?)
}

fn hex(bytes: [u8; 32]) -> Result<String> {
    let mut hex_bytes = [0_u8; 64];
    hex::encode_to_slice(bytes, &mut hex_bytes).map_err(|_| reject("sha256 digest length"))?;
    let mut result = String::new();
    result.try_reserve_exact(hex_bytes.len())?;
    for byte in hex_bytes {
        result.push(char::from(byte));
    }
    Ok(result)
}

fn path_basename(path: &Path) -> String {
    path.file_name().map_or_else(
        || String::from("capture.json"),
        |name| name.to_string_lossy().into_owned(),
    )
}

fn bytes_sha256(bytes: &[u8]) -> Result<String> {
    let mut hash = Hasher::new()?;
    hash.update(bytes)?;
    hex(hash.finish()?)
}

fn capture_artifact_summary(role: &str, path: &Path, bytes: &[u8]) -> Result<Value> {
    Ok(json!({
        "role": role,
        "name": path_basename(path),
        "bytes": bytes.len(),
        "sha256": bytes_sha256(bytes)?,
    }))
}

fn ensure_finite(values: &[f32]) -> Result<()> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(reject("FFN probe vector contains non-finite values"));
    }
    Ok(())
}

fn sum_of_squares(values: &[f32]) -> Result<f64> {
    ensure_finite(values)?;
    let mut sum_sq = 0.0_f64;
    for &value in values {
        let wide = f64::from(value);
        sum_sq = wide.mul_add(wide, sum_sq);
    }
    if !sum_sq.is_finite() {
        return Err(reject("FFN probe vector energy overflowed"));
    }
    Ok(sum_sq)
}

fn l2_norm(values: &[f32]) -> Result<f64> {
    let norm = sum_of_squares(values)?.sqrt();
    if !norm.is_finite() {
        return Err(reject("FFN probe vector norm overflowed"));
    }
    Ok(norm)
}

fn cosine_similarity(left: &[f32], right: &[f32]) -> Result<Option<f64>> {
    ensure_finite(left)?;
    ensure_finite(right)?;
    if left.len() != right.len() {
        return Err(reject("cosine similarity requires equal vector lengths"));
    }
    let left_norm = l2_norm(left)?;
    let right_norm = l2_norm(right)?;
    if left_norm <= 0.0 || right_norm <= 0.0 {
        return Ok(None);
    }
    let mut dot = 0.0_f64;
    for (&lhs, &rhs) in left.iter().zip(right.iter()) {
        dot = f64::from(lhs).mul_add(f64::from(rhs), dot);
    }
    let ratio = (dot / (left_norm * right_norm)).clamp(-1.0, 1.0);
    if !ratio.is_finite() {
        return Err(reject("cosine similarity produced non-finite value"));
    }
    Ok(Some(ratio))
}

fn energy_entropy(values: &[f32]) -> Result<f64> {
    let total_energy = sum_of_squares(values)?;
    if total_energy <= 0.0 {
        return Ok(0.0);
    }
    let mut entropy = 0.0_f64;
    for &value in values {
        let wide = f64::from(value);
        let energy = wide * wide;
        if energy > 0.0 {
            let probability = energy / total_energy;
            entropy = (-probability).mul_add(probability.ln(), entropy);
        }
    }
    if !entropy.is_finite() {
        return Err(reject("FFN probe vector entropy overflowed"));
    }
    Ok(if entropy <= 0.0 { 0.0 } else { entropy })
}

fn coordinate_coverage(values: &[f32]) -> Result<f64> {
    ensure_finite(values)?;
    if values.is_empty() {
        return Ok(0.0);
    }
    let nonzero = values.iter().filter(|&&value| value.abs() > 0.0).count();
    let numerator = f64::from(u32::try_from(nonzero)?);
    let denominator = f64::from(u32::try_from(values.len())?);
    Ok(numerator / denominator)
}

struct VectorStats {
    l2_norm: f64,
    energy_entropy: f64,
    coordinate_coverage: f64,
}

impl VectorStats {
    fn from_slice(values: &[f32]) -> Result<Self> {
        Ok(Self {
            l2_norm: l2_norm(values)?,
            energy_entropy: energy_entropy(values)?,
            coordinate_coverage: coordinate_coverage(values)?,
        })
    }

    fn to_json(&self) -> Value {
        json!({
            "l2Norm": self.l2_norm,
            "energyEntropy": self.energy_entropy,
            "coordinateCoverage": self.coordinate_coverage,
        })
    }
}

#[derive(Default)]
struct StatsAccumulator {
    count: u32,
    l2_norm_sum: f64,
    energy_entropy_sum: f64,
    coordinate_coverage_sum: f64,
}

impl StatsAccumulator {
    fn push(&mut self, stats: &VectorStats) -> Result<()> {
        self.count = self
            .count
            .checked_add(1)
            .ok_or_else(|| reject("FFN probe row count overflow"))?;
        self.l2_norm_sum += stats.l2_norm;
        self.energy_entropy_sum += stats.energy_entropy;
        self.coordinate_coverage_sum += stats.coordinate_coverage;
        if !self.l2_norm_sum.is_finite()
            || !self.energy_entropy_sum.is_finite()
            || !self.coordinate_coverage_sum.is_finite()
        {
            return Err(reject("FFN probe aggregate vector statistics overflowed"));
        }
        Ok(())
    }

    fn summary(&self) -> Value {
        if self.count == 0 {
            return json!({
                "count": 0,
                "meanL2Norm": Value::Null,
                "meanEnergyEntropy": Value::Null,
                "meanCoordinateCoverage": Value::Null,
            });
        }
        let denom = f64::from(self.count);
        json!({
            "count": self.count,
            "meanL2Norm": self.l2_norm_sum / denom,
            "meanEnergyEntropy": self.energy_entropy_sum / denom,
            "meanCoordinateCoverage": self.coordinate_coverage_sum / denom,
        })
    }
}

fn row_pair_stats(
    first_name: &str,
    first: &[f32],
    second_name: &str,
    second: &[f32],
    include_cosine: bool,
) -> Result<(VectorStats, VectorStats, Option<f64>, Value)> {
    let first_stats = VectorStats::from_slice(first)?;
    let second_stats = VectorStats::from_slice(second)?;
    let cosine = if include_cosine {
        cosine_similarity(first, second)?
    } else {
        None
    };
    let mut value = json!({
        first_name: first_stats.to_json(),
        second_name: second_stats.to_json(),
    });
    if include_cosine {
        value["inputTargetCosine"] = json!(cosine);
    }
    Ok((first_stats, second_stats, cosine, value))
}

fn parse_scale(bits: &str) -> Result<f32> {
    if bits.len() != 8 || !bits.bytes().all(lower_hex) {
        return Err(reject(
            "adapterScaleBits must contain eight hexadecimal digits",
        ));
    }
    let scale = f32::from_bits(u32::from_str_radix(bits, 16)?);
    if !scale.is_finite() || !(0.0..=2.0).contains(&scale) {
        return Err(reject("adapter scale must be finite and in 0..2"));
    }
    Ok(scale)
}

const fn lower_hex(value: u8) -> bool {
    value.is_ascii_digit() || matches!(value, b'a'..=b'f')
}

fn identifier(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 128
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
}

fn validate(request: &Request) -> Result<Representation> {
    if request.schema_version != 1
        || !request.model_path.is_absolute()
        || !request.capture_path.is_absolute()
        || request.capture_path.exists()
        || request.runtime_source_commit.len() != 40
        || !request.runtime_source_commit.bytes().all(lower_hex)
        || request.layer >= 64
        || request.sequences.is_empty()
        || request.sequences.len() > 16
        || (request.capture_kind == CaptureKind::FfnSurrogate && request.patch_path.is_some())
        || match (&request.omitted_groups, request.capture_kind) {
            (Some(groups), CaptureKind::FfnGroupOmission) => {
                groups.is_empty()
                    || groups.len() >= 17
                    || groups.iter().any(|&group| group >= 17)
                    || groups.windows(2).any(|pair| pair[0] >= pair[1])
            }
            (None, CaptureKind::FfnGroupOmission) | (Some(_), _) => true,
            (None, _) => false,
        }
    {
        return Err(reject(
            "invalid FFN probe version, paths, source, site or sequences",
        ));
    }
    let representation = match request.weight_representation.as_str() {
        "PQ2_0" => Representation::Pq2,
        "current-repacked-PTQ1_0" => Representation::CurrentRepackedPtq1,
        _ => return Err(reject("unsupported current-weight representation")),
    };
    let mut tokens = 0_usize;
    let mut rows = 0_usize;
    let mut samples = BTreeSet::new();
    let mut groups = std::collections::BTreeMap::new();
    let mut fit_rows = 0;
    let mut development_rows = 0;
    for sequence in &request.sequences {
        tokens = tokens
            .checked_add(sequence.token_ids.len())
            .ok_or_else(|| reject("token count overflow"))?;
        rows = rows
            .checked_add(sequence.capture_token_indices.len())
            .ok_or_else(|| reject("capture count overflow"))?;
        let indices = sequence
            .capture_token_indices
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        if !identifier(&sequence.sample_id)
            || !identifier(&sequence.group_id)
            || !samples.insert(&sequence.sample_id)
            || match request.capture_kind {
                CaptureKind::HiddenPatch => {
                    !matches!(sequence.split.as_str(), "fit" | "holdout")
                        || !matches!(sequence.label.as_deref(), Some("target" | "preserve"))
                }
                CaptureKind::FfnSurrogate | CaptureKind::FfnGroupOmission => {
                    !matches!(sequence.split.as_str(), "fit" | "development")
                }
            }
            || sequence.token_ids.is_empty()
            || sequence.capture_token_indices.is_empty()
            || indices.len() != sequence.capture_token_indices.len()
            || indices
                .iter()
                .any(|&index| index >= sequence.token_ids.len())
            || sequence.token_ids.iter().any(|&token| {
                usize::try_from(token).map_or(true, |token| token >= swift::VOCAB_SIZE)
            })
            || groups
                .insert(&sequence.group_id, &sequence.split)
                .is_some_and(|previous| previous != &sequence.split)
        {
            return Err(reject(
                "invalid FFN probe sequence identity, split, label or token bounds",
            ));
        }
        if sequence.split == "fit" {
            fit_rows += sequence.capture_token_indices.len();
        } else if sequence.split == "development" {
            development_rows += sequence.capture_token_indices.len();
        }
    }
    if tokens > MAX_TOKENS
        || rows > MAX_CAPTURE_ROWS
        || (request.capture_kind == CaptureKind::FfnGroupOmission && tokens > MAX_TOKENS / 2)
        || (request.capture_kind != CaptureKind::HiddenPatch
            && (fit_rows < 2 || development_rows == 0))
    {
        return Err(reject(
            "FFN probe exceeds aggregate token or capture bounds",
        ));
    }
    Ok(representation)
}

#[expect(
    clippy::too_many_lines,
    reason = "Keep exact-source intake, model construction, provisional capture and publication order auditable."
)]
fn run() -> Result<Value> {
    let started_wall = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis();
    let started = Instant::now();
    let arguments = std::env::args_os().skip(1).collect::<Vec<_>>();
    if arguments.len() != 1 {
        return Err(reject(
            "usage: ffn_probe <absolute-request.json>; external Job/deadline/power supervision required",
        ));
    }
    let request: Request =
        serde_json::from_slice(&read_bounded(Path::new(&arguments[0]), MAX_REQUEST_BYTES)?)?;
    let remaining = u128::from(request.absolute_deadline_unix_ms)
        .checked_sub(started_wall)
        .filter(|remaining| (1..=240_000).contains(remaining))
        .ok_or_else(|| {
            reject("original absolute deadline must leave at most 240 seconds at entry")
        })?;
    let deadline = Deadline(started + Duration::from_millis(u64::try_from(remaining)?));
    deadline.check()?;
    let representation = validate(&request)?;
    let scale = parse_scale(&request.adapter_scale_bits)?;
    if request.adapter_path.is_none()
        && (scale != 0.0
            || (request.capture_kind != CaptureKind::HiddenPatch && scale.to_bits() != 0))
    {
        return Err(reject("absent adapter requires zero scale"));
    }
    let executable_sha256 = hash_file(&std::env::current_exe()?, &deadline)?;
    deadline.check()?;
    let (model, _) = swift::load_representation(&request.model_path, MODEL_BUDGET, representation)?;
    deadline.check()?;
    let context = request
        .sequences
        .iter()
        .map(|sequence| sequence.token_ids.len())
        .max()
        .ok_or_else(|| reject("missing sequences"))?;
    let graph = if let Some(path) = &request.adapter_path {
        if !path.is_absolute() {
            return Err(reject("adapter path must be absolute"));
        }
        let adapter = RankOneAdapter::load_swift_representation(path, &model, representation)?;
        Qwen35::with_rank_one_adapter(
            model,
            context,
            MAX_STATE_BYTES,
            MatvecBackend::Scalar,
            adapter,
            scale,
        )?
    } else {
        Qwen35::with_backend(model, context, MatvecBackend::Scalar)?
    };
    if request.capture_kind == CaptureKind::FfnGroupOmission {
        return group_omission::capture(
            &request,
            representation,
            scale,
            &executable_sha256,
            graph,
            &deadline,
        );
    }
    let base_probe_identity = format!(
        "{}:{EXECUTION_PROFILE}:layer={}:patch=none",
        graph.identity(),
        request.layer
    );
    let mut source = json!({
        "modelSha256":representation.model_sha256(),
        "runtimeSha256":executable_sha256,
        "adapterSha256":request.adapter_path.as_ref().map(|_| SWIFT_ADAPTER_SHA256),
        "adapterScaleBits":format!("{:08x}", scale.to_bits()),
        "weightRepresentation":representation.name(),
        "executionProfile":EXECUTION_PROFILE,
        "runtimeSourceCommit":request.runtime_source_commit,
        "graphIdentity":base_probe_identity,
    });
    let site = match request.capture_kind {
        CaptureKind::HiddenPatch => json!({
            "tensor":format!("blk.{}.ffn_down.weight", request.layer),
            "inputBasis":INPUT_BASIS,"inputWidth":17408,"outputWidth":5120,
        }),
        CaptureKind::FfnSurrogate => json!({
            "layer":request.layer,"inputBasis":SURROGATE_INPUT_BASIS,
            "targetBasis":SURROGATE_TARGET_BASIS,"inputWidth":5120,"outputWidth":5120,
        }),
        CaptureKind::FfnGroupOmission => {
            return Err(reject("structured capture must use its two-pass path"));
        }
    };
    let patch = request
        .patch_path
        .as_ref()
        .map(|path| read_patch(path, &source, &site))
        .transpose()?;
    let mut graph = FfnProbe::new(graph, request.layer, patch)?;
    // Observations must identify the graph that actually produced them. A
    // later fit on patched captures cannot masquerade as an unpatched fit.
    source["graphIdentity"] = graph.identity().into();
    let mut rows = Vec::new();
    let mut logits = Vec::new();
    let mut first_accumulator = StatsAccumulator::default();
    let mut second_accumulator = StatsAccumulator::default();
    let mut cosine_count = 0_u32;
    let mut cosine_sum = 0.0_f64;
    for sequence in &request.sequences {
        graph.reset()?;
        for (index, &token) in sequence.token_ids.iter().enumerate() {
            deadline.check()?;
            if request.capture_kind == CaptureKind::FfnSurrogate {
                if sequence.capture_token_indices.contains(&index) {
                    let observation = graph.evaluate_surrogate_into(token, &mut logits)?;
                    if observation.input.len() != 5120 || observation.target.len() != 5120 {
                        return Err(reject(
                            "actual surrogate observation dimensions differ from Swift contract",
                        ));
                    }
                    deadline.check()?;
                    let (input_stats, target_stats, cosine, pair_json) = row_pair_stats(
                        "input",
                        &observation.input,
                        "target",
                        &observation.target,
                        true,
                    )?;
                    first_accumulator.push(&input_stats)?;
                    second_accumulator.push(&target_stats)?;
                    if let Some(cosine) = cosine {
                        cosine_count = cosine_count
                            .checked_add(1)
                            .ok_or_else(|| reject("FFN probe cosine count overflow"))?;
                        cosine_sum += cosine;
                    }
                    rows.push(json!({
                        "sampleId":sequence.sample_id,"groupId":sequence.group_id,
                        "tokenIndex":index,"split":sequence.split,
                        "input":observation.input,"target":observation.target,
                        "vectorStats":pair_json,
                    }));
                } else {
                    graph.evaluate_into(token, &mut logits)?;
                }
                continue;
            }
            let mut row = None;
            graph.evaluate_observed_into(token, &mut logits, &mut |observation| {
                if sequence.capture_token_indices.contains(&index) {
                    if observation.activation.len() != 17408 || observation.residual.len() != 5120 {
                        return Err(reject(
                            "actual FFN observation dimensions differ from Swift contract",
                        ));
                    }
                    let (act_stats, res_stats, _, pair_json) = row_pair_stats(
                        "activation",
                        observation.activation,
                        "residual",
                        observation.residual,
                        false,
                    )?;
                    row = Some((
                        json!({
                            "sampleId":sequence.sample_id,"groupId":sequence.group_id,
                            "tokenIndex":index,"split":sequence.split,"label":sequence.label,
                            "activation":observation.activation,"residual":observation.residual,
                            "vectorStats":pair_json,
                        }),
                        act_stats,
                        res_stats,
                    ));
                }
                Ok(())
            })?;
            deadline.check()?;
            if let Some((row, act_stats, res_stats)) = row {
                first_accumulator.push(&act_stats)?;
                second_accumulator.push(&res_stats)?;
                rows.push(row);
            }
        }
    }
    let row_count = rows.len();
    let artifact = match request.capture_kind {
        CaptureKind::HiddenPatch => json!({
            "schemaVersion":1,"source":source,"site":site,"rows":rows,
            "vectorStats":{
                "activation":first_accumulator.summary(),
                "residual":second_accumulator.summary(),
            },
            "observationGraphIdentity":graph.identity(),
            "qualification":"teacher-forced observations only; labels are supplied, not independently diagnosed; no behavioral or performance acceptance",
        }),
        CaptureKind::FfnSurrogate => {
            let mean_cosine = if cosine_count == 0 {
                Value::Null
            } else {
                json!(cosine_sum / f64::from(cosine_count))
            };
            json!({
                "schemaVersion":1,"source":source,"site":site,"rows":rows,
                "vectorStats":{
                    "input":first_accumulator.summary(),
                    "target":second_accumulator.summary(),
                    "cosineSampleCount":cosine_count,
                    "meanInputTargetCosine":mean_cosine,
                },
            })
        }
        CaptureKind::FfnGroupOmission => {
            return Err(reject("structured capture must use its two-pass path"));
        }
    };
    let bytes = serde_json::to_vec(&artifact)?;
    if bytes.len() > MAX_CAPTURE_BYTES {
        return Err(reject("serialized FFN capture exceeds its byte bound"));
    }
    deadline.check()?;
    publish(&request.capture_path, &bytes)?;
    Ok(json!({
        "event": "ffn-probe-captured",
        "rows": row_count,
        "captureArtifact": capture_artifact_summary("ffn-capture", &request.capture_path, &bytes)?,
        "executionProfile": EXECUTION_PROFILE,
        "qualification": "unqualified research capture",
    }))
}

fn factor(value: &Value, expected: usize) -> Result<Vec<f32>> {
    let values = value
        .as_array()
        .filter(|values| values.len() == expected)
        .ok_or_else(|| reject("patch factor shape mismatch"))?;
    values
        .iter()
        .map(|value| {
            let value: f32 = serde_json::from_value(value.clone())?;
            if !value.is_finite() {
                return Err(reject("nonfinite patch factor"));
            }
            Ok(value)
        })
        .collect()
}

fn factor_hash(values: &[f32]) -> Result<String> {
    let mut hash = Hasher::new()?;
    for value in values {
        hash.update(&value.to_le_bytes())?;
    }
    hex(hash.finish()?)
}

fn read_patch(artifact_path: &Path, source: &Value, site: &Value) -> Result<FfnPatch> {
    let patch: Value = serde_json::from_slice(&read_bounded(artifact_path, MAX_PATCH_BYTES)?)?;
    if patch["schemaVersion"] != 1
        || patch["source"] != *source
        || patch["site"] != *site
        || patch["scaleBits"] != "3f800000"
        || patch["kind"] != "experimental-ffn-rank-one-patch"
        || patch["compositionOrder"] != "after-standard-adapter"
        || patch["enabled"] != false
        || patch["behaviorQualified"] != false
        || patch["qualification"] != "unqualified-research-only"
        || patch["optimizationCertificate"]["certified"] != true
        || patch["optimizationCertificate"]["evaluatedFactors"] != "F32-le"
    {
        return Err(reject("patch source/site/scale does not match exact probe"));
    }
    let gap = patch["optimizationCertificate"]["relativeGap"]
        .as_f64()
        .filter(|value| value.is_finite() && *value >= 0.0)
        .ok_or_else(|| reject("invalid patch optimization gap"))?;
    let tolerance = patch["optimizationCertificate"]["relativeGapTolerance"]
        .as_f64()
        .filter(|value| value.is_finite() && *value > 0.0)
        .ok_or_else(|| reject("invalid patch optimization tolerance"))?;
    if gap > tolerance {
        return Err(reject("patch did not meet its optimization tolerance"));
    }
    let a = factor(&patch["factors"]["a"], 17408)?;
    let b = factor(&patch["factors"]["b"], 5120)?;
    if patch["factors"]["aSha256"] != factor_hash(&a)?
        || patch["factors"]["bSha256"] != factor_hash(&b)?
    {
        return Err(reject(
            "patch factor hashes do not match exact binary32 payload",
        ));
    }
    FfnPatch::new(a, b, 1.0)
}

fn publish(path: &Path, bytes: &[u8]) -> Result<()> {
    let temporary = path.with_file_name(format!(
        ".ffn-capture-{}-{}.pending",
        std::process::id(),
        SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos()
    ));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)?;
    let result: Result<()> = (|| {
        file.write_all(bytes)?;
        file.sync_all()?;
        drop(file);
        // Same-directory hard-link publication is atomic and refuses an
        // existing destination on every supported platform.
        fs::hard_link(&temporary, path)?;
        Ok(())
    })();
    let cleanup = fs::remove_file(&temporary);
    result?;
    cleanup?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{CaptureKind, Request, parse_scale, validate};
    use bonsai_prism_runtime::Result;
    use serde_json::json;

    fn request() -> Result<Request> {
        let directory = std::env::temp_dir();
        Ok(serde_json::from_value(json!({
            "schemaVersion":1,"modelPath":directory.join("ffn-probe-model.gguf"),
            "adapterPath":null,"adapterScaleBits":"00000000","weightRepresentation":"PQ2_0",
            "layer":0,"runtimeSourceCommit":"1".repeat(40),"absoluteDeadlineUnixMs":1,
            "capturePath":directory.join(format!("ffn-probe-control-{}.json",std::process::id())),
            "patchPath":null,"sequences":[
                {"sampleId":"fit-target","groupId":"fit-trace","split":"fit","label":"target","tokenIds":[1,2],"captureTokenIndices":[1]},
                {"sampleId":"holdout-preserve","groupId":"holdout-trace","split":"holdout","label":"preserve","tokenIds":[1],"captureTokenIndices":[0]}
            ]
        }))?)
    }

    #[test]
    fn ffn_capture_rejects_trace_leakage_and_invalid_token_selection() -> Result<()> {
        let mut request = request()?;
        assert!(validate(&request).is_ok());
        request.sequences[1].group_id = request.sequences[0].group_id.clone();
        assert!(validate(&request).is_err());
        request.sequences[1].group_id = "holdout-trace".into();
        request.sequences[0].capture_token_indices = vec![1, 1];
        assert!(validate(&request).is_err());
        request.sequences[0].capture_token_indices = vec![2];
        assert!(validate(&request).is_err());
        Ok(())
    }

    #[test]
    fn ffn_capture_scale_bits_are_canonical_finite_and_bounded() -> Result<()> {
        assert_eq!(parse_scale("3fa66666")?, 1.3);
        for bits in ["3FA66666", "7f800000", "7fc00000", "40000001", "bf800000"] {
            assert!(parse_scale(bits).is_err());
        }
        Ok(())
    }

    #[test]
    fn ffn_surrogate_capture_keeps_development_separate_and_rejects_patches() -> Result<()> {
        let mut request = request()?;
        request.capture_kind = CaptureKind::FfnSurrogate;
        request.sequences[0].capture_token_indices = vec![0, 1];
        request.sequences[1].split = "development".into();
        for sequence in &mut request.sequences {
            sequence.label = None;
        }
        assert!(validate(&request).is_ok());
        request.sequences[1].split = "holdout".into();
        assert!(validate(&request).is_err());
        request.sequences[1].split = "development".into();
        request.patch_path = Some(std::env::temp_dir().join("independent-patch.json"));
        assert!(validate(&request).is_err());
        request.patch_path = None;
        request.sequences[0].capture_token_indices = vec![1];
        assert!(validate(&request).is_err());
        Ok(())
    }

    #[test]
    fn structured_capture_rejects_group_errors_and_charges_both_replays() -> Result<()> {
        let mut request = request()?;
        request.capture_kind = CaptureKind::FfnGroupOmission;
        request.omitted_groups = Some(vec![0, 16]);
        request.sequences[0].capture_token_indices = vec![0, 1];
        request.sequences[1].split = "development".into();
        for sequence in &mut request.sequences {
            sequence.label = None;
        }
        assert!(validate(&request).is_ok());
        for groups in [
            None,
            Some(vec![]),
            Some(vec![1, 1]),
            Some(vec![16, 0]),
            Some(vec![17]),
            Some((0..17).collect()),
        ] {
            request.omitted_groups = groups;
            assert!(validate(&request).is_err());
        }
        request.omitted_groups = Some(vec![0, 16]);
        request.sequences[0].token_ids = vec![1; 127];
        assert!(validate(&request).is_ok());
        request.sequences[0].token_ids.push(1);
        assert!(validate(&request).is_err());
        let _ = request.sequences[0].token_ids.pop();
        request.patch_path = Some(std::env::temp_dir().join("unsupported-recovery.json"));
        assert!(validate(&request).is_ok());
        request.patch_path = None;
        request.capture_kind = CaptureKind::FfnSurrogate;
        assert!(validate(&request).is_err());
        Ok(())
    }

    #[test]
    fn vector_telemetry_rejects_non_finite_and_mismatched_and_matches_exact_identities()
    -> Result<()> {
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(super::l2_norm(&[1.0, bad]).is_err());
            assert!(super::energy_entropy(&[1.0, bad]).is_err());
            assert!(super::coordinate_coverage(&[1.0, bad]).is_err());
            assert!(super::cosine_similarity(&[1.0, 2.0], &[1.0, bad]).is_err());
        }
        assert!(super::cosine_similarity(&[1.0, 2.0], &[1.0]).is_err());

        assert!(super::l2_norm(&[0.0, -0.0, 0.0])?.abs() < 1e-12);
        assert!(super::energy_entropy(&[0.0, -0.0])?.abs() < 1e-12);
        assert!(super::coordinate_coverage(&[0.0, -0.0, 0.0])?.abs() < 1e-12);
        assert!(super::cosine_similarity(&[0.0, 0.0], &[1.0, 0.0])?.is_none());

        assert!((super::l2_norm(&[3.0, -4.0])? - 5.0).abs() < 1e-12);
        assert!(super::energy_entropy(&[0.0, 5.0, 0.0])?.abs() < 1e-12);
        assert!((super::energy_entropy(&[2.0, -2.0, 2.0, -2.0])? - 4.0_f64.ln()).abs() < 1e-12);
        assert!((super::coordinate_coverage(&[0.0, 1.5, -0.0, -2.0])? - 0.5).abs() < 1e-12);

        let same = super::cosine_similarity(&[1.0, -2.0], &[3.0, -6.0])?.ok_or("missing cosine")?;
        assert!((same - 1.0).abs() < 1e-12);
        let opposite =
            super::cosine_similarity(&[1.0, 2.0], &[-1.0, -2.0])?.ok_or("missing cosine")?;
        assert!((opposite + 1.0).abs() < 1e-12);
        let orthogonal =
            super::cosine_similarity(&[1.0, 0.0], &[0.0, 2.0])?.ok_or("missing cosine")?;
        assert!(orthogonal.abs() < 1e-12);

        let (_, _, cosine, stats_json) =
            super::row_pair_stats("input", &[3.0, 0.0], "target", &[0.0, 4.0], true)?;
        assert!(cosine.ok_or("missing pair cosine")?.abs() < 1e-12);
        assert!(
            (stats_json["input"]["l2Norm"]
                .as_f64()
                .ok_or("missing norm")?
                - 3.0)
                .abs()
                < 1e-12
        );
        assert!(
            (stats_json["target"]["coordinateCoverage"]
                .as_f64()
                .ok_or("missing coverage")?
                - 0.5)
                .abs()
                < 1e-12
        );
        assert_eq!(
            super::path_basename(std::path::Path::new("C:\\private\\dir\\capture.json")),
            "capture.json"
        );
        assert_eq!(
            super::bytes_sha256(b"")?,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        Ok(())
    }
}
