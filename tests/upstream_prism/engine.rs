//! Production operation adapters; inputs and assertions remain in Rust.

use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use bonsai_prism_runtime::kernels::{matvec, row};
use bonsai_prism_runtime::model::Tensor;
use serde_json::Value;

use super::fixture::Fixture;
use super::{Result, invalid};

const MAX_RESPONSE_BYTES: u64 = 16 * 1024 * 1024;
const MAX_LOG_BYTES: u64 = 1024 * 1024;
// Leave the supervising runner time to collect evidence and reap the harness.
const BRIDGE_TIMEOUT: Duration = Duration::from_secs(90);

#[derive(Clone, Copy)]
pub(super) enum Target {
    Cpp,
    Rust,
}

pub(super) struct Pq2Input {
    pub(super) k: usize,
    pub(super) m: usize,
    pub(super) n: usize,
    pub(super) packed: Vec<u8>,
    pub(super) activations: Vec<f32>,
}

pub(super) enum Request {
    Pq2(Pq2Input),
    Ptq1(Vec<[u8; 28]>),
}

type Outcome = std::result::Result<Vec<f32>, String>;

pub(super) enum Output {
    Pq2(Outcome),
    Ptq1(Vec<Outcome>),
}

pub(super) struct Engine {
    target: Target,
    bridge: Option<PathBuf>,
    common_bridge: Option<PathBuf>,
    evidence_directory: Option<PathBuf>,
    source_directory: Option<PathBuf>,
    asset_directory: Option<PathBuf>,
}

pub(super) struct ModelSessionSpec {
    pub(super) executable: PathBuf,
    pub(super) evidence_directory: PathBuf,
}

impl Engine {
    pub(super) const fn is_rust(&self) -> bool {
        matches!(self.target, Target::Rust)
    }

    pub(super) fn new(
        target: Target,
        bridge: Option<PathBuf>,
        common_bridge: Option<PathBuf>,
        evidence_directory: Option<PathBuf>,
        source_directory: Option<PathBuf>,
        asset_directory: Option<PathBuf>,
    ) -> Result<Self> {
        for path in [&bridge, &common_bridge].into_iter().flatten() {
            if !path.is_absolute() || !path.is_file() {
                return Err(invalid("native bridge must be an existing absolute file"));
            }
        }
        if let Some(path) = &evidence_directory {
            if !path.is_absolute() {
                return Err(invalid("bridge evidence directory must be absolute"));
            }
            fs::create_dir_all(path)?;
        }
        let engine = Self {
            target,
            bridge,
            common_bridge,
            evidence_directory,
            source_directory,
            asset_directory,
        };
        if engine.source_directory.is_some() {
            let root = engine.source_path("")?;
            if !root.is_absolute() || !root.is_dir() {
                return Err(invalid(
                    "upstream source directory must be an existing absolute directory",
                ));
            }
        }
        if engine.asset_directory.is_some() {
            let root = engine.asset_path("")?;
            if !root.is_absolute() || !root.is_dir() {
                return Err(invalid(
                    "model asset directory must be an existing absolute directory",
                ));
            }
        }
        Ok(engine)
    }

    pub(super) fn source_path(&self, relative: &str) -> Result<PathBuf> {
        Self::relative_path(
            self.source_directory.as_deref(),
            relative,
            "--source-directory",
        )
    }

    pub(super) fn asset_path(&self, relative: &str) -> Result<PathBuf> {
        Self::relative_path(
            self.asset_directory.as_deref(),
            relative,
            "--asset-directory",
        )
    }

    fn relative_path(root: Option<&Path>, relative: &str, option: &str) -> Result<PathBuf> {
        let path = Path::new(relative);
        if path
            .components()
            .any(|component| !matches!(component, std::path::Component::Normal(_)))
        {
            return Err(invalid(
                "upstream asset path must contain only relative normal components",
            ));
        }
        Ok(root
            .ok_or_else(|| invalid(format!("selected source requires {option}")))?
            .join(path))
    }

    pub(super) fn evidence_path(&self, name: &str) -> Result<PathBuf> {
        if name.is_empty()
            || !name
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-')
        {
            return Err(invalid("invalid native evidence name"));
        }
        let evidence = self
            .evidence_directory
            .as_ref()
            .ok_or_else(|| invalid("native bridge evidence directory is missing"))?;
        let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let directory = evidence.join(format!("{name}-{}-{nonce}", std::process::id()));
        fs::create_dir(&directory)?;
        Ok(directory)
    }

    pub(super) fn model_session_spec(&self, name: &str) -> Result<Option<ModelSessionSpec>> {
        if matches!(self.target, Target::Rust) {
            return Ok(None);
        }
        Ok(Some(ModelSessionSpec {
            executable: self
                .common_bridge
                .clone()
                .ok_or_else(|| invalid("native common bridge path is missing"))?,
            evidence_directory: self.evidence_path(
                &name
                    .chars()
                    .map(|character| {
                        if character.is_ascii_alphanumeric() {
                            character
                        } else {
                            '-'
                        }
                    })
                    .collect::<String>(),
            )?,
        }))
    }

    pub(super) fn production_tool(&self, name: &str) -> Result<Option<PathBuf>> {
        if name != "llama-export-lora" {
            return Err(invalid("unknown production tool"));
        }
        if matches!(self.target, Target::Rust) {
            return Ok(None);
        }
        let bridge = fs::canonicalize(
            self.common_bridge
                .as_ref()
                .ok_or_else(|| invalid("native common bridge missing"))?,
        )?;
        let directory = bridge
            .parent()
            .ok_or_else(|| invalid("native bridge has no parent"))?;
        let file = directory.join(if cfg!(windows) {
            "llama-export-lora.exe"
        } else {
            "llama-export-lora"
        });
        if !file.is_file() {
            return Err(invalid("verified production tool was not provisioned"));
        }
        let file = fs::canonicalize(file)?;
        if file.parent() != Some(directory) {
            return Err(invalid(
                "production tool escaped the verified bridge directory",
            ));
        }
        Ok(Some(file))
    }

    pub(super) fn execute(&self, name: &str, requests: &[Request]) -> Result<Vec<Output>> {
        match self.target {
            Target::Cpp => self.cpp(name, requests),
            Target::Rust => Ok(requests.iter().map(rust_operation).collect()),
        }
    }

    fn cpp(&self, name: &str, requests: &[Request]) -> Result<Vec<Output>> {
        let bridge = self
            .bridge
            .as_ref()
            .ok_or_else(|| invalid("native bridge path is missing"))?;
        decode_response(
            &self.exchange(
                bridge,
                name,
                &encode_requests(requests)?,
                "bin",
                BRIDGE_TIMEOUT,
            )?,
            requests,
        )
    }

    // None denotes a missing Rust production API. The suite must report that
    // as a failed contract, never substitute a test-only implementation.
    pub(super) fn execute_common(&self, name: &str, request: &Value) -> Result<Option<Value>> {
        self.execute_common_with_timeout(name, request, BRIDGE_TIMEOUT)
    }

    pub(super) fn execute_common_with_timeout(
        &self,
        name: &str,
        request: &Value,
        timeout: Duration,
    ) -> Result<Option<Value>> {
        if timeout.is_zero() || timeout > Duration::from_secs(3600) {
            return Err(invalid(
                "common bridge timeout must be positive and at most one hour",
            ));
        }
        let backend_request = request.as_object().is_some_and(|object| object.len() == 2)
            && request["SchemaVersion"] == 1
            && request["Operations"].as_array().is_some_and(|operations| {
                operations.len() == 1 && operations[0]["Kind"] == "backend-graph"
            });
        if timeout > BRIDGE_TIMEOUT
            && (!backend_request
                || std::env::var("GITHUB_ACTIONS").as_deref() != Ok("true")
                || std::env::var("GITHUB_REF").as_deref() != Ok("refs/heads/main"))
        {
            return Err(invalid(
                "extended graph-controller deadlines require the main CI backend lane",
            ));
        }
        if matches!(self.target, Target::Rust) {
            if let Some(response) = super::engine_schema::execute(request)? {
                return Ok(Some(response));
            }
            if let Some(response) = super::engine_unicode::execute(request)? {
                return Ok(Some(response));
            }
            if let Some(response) = super::mtmd_chunks::execute_runtime(request)? {
                return Ok(Some(response));
            }
            if let Some(response) = super::peg::execute(request)? {
                return Ok(Some(response));
            }
            if let Some(response) = super::gguf::execute_runtime(request)? {
                return Ok(Some(response));
            }
            if let Some(response) = super::quant_selection::execute_runtime(request)? {
                return Ok(Some(response));
            }
            if let Some(response) = super::optimizer::execute_runtime(request)? {
                return Ok(Some(response));
            }
            if let Some(response) = super::engine_runtime::execute(request)? {
                return Ok(Some(response));
            }
            if let Some(response) = super::reasoning_budget::execute_runtime(request)? {
                return Ok(Some(response));
            }
            if let Some(response) = super::sampling::execute_runtime(request)? {
                return Ok(Some(response));
            }
            if let Some(response) = super::batch_allocator::execute_runtime(request)? {
                return Ok(Some(response));
            }
            if let Some(response) = super::backend_ops::execute(request)? {
                return Ok(Some(response));
            }
            if let Some(response) = super::llama_grammar::execute_runtime(request)? {
                return Ok(Some(response));
            }
            if let Some(response) = super::jinja::execute(request)? {
                return Ok(Some(response));
            }
            if let Some(response) = super::chat_template::execute(request)? {
                return Ok(Some(response));
            }
            if let Some(response) = super::chat_peg::execute_operations(request)? {
                return Ok(Some(response));
            }
            return super::engine_vocab::execute(request);
        }
        let bridge = self
            .common_bridge
            .as_ref()
            .ok_or_else(|| invalid("native common bridge path is missing"))?;
        let request = serde_json::to_vec(request)?;
        if request.len() > 16 * 1024 * 1024 {
            return Err(invalid("common bridge request exceeds size bound"));
        }
        Ok(Some(serde_json::from_slice(
            &self.exchange(bridge, name, &request, "json", timeout)?,
        )?))
    }

    pub(super) fn execute_codec(&self, name: &str, request: &Value) -> Result<Option<Value>> {
        match self.target {
            Target::Cpp => self.execute_common(name, request),
            Target::Rust => super::engine_codec::execute(request),
        }
    }

    fn exchange(
        &self,
        bridge: &Path,
        name: &str,
        request: &[u8],
        extension: &str,
        timeout: Duration,
    ) -> Result<Vec<u8>> {
        let directory = self.evidence_path(name)?;
        let input = directory.join(format!("request.{extension}"));
        let output = directory.join(format!("response.{extension}"));
        fs::write(&input, request)?;
        run_bridge(bridge, &directory, &input, &output, timeout)?;
        if fs::metadata(&output)?.len() > MAX_RESPONSE_BYTES {
            return Err(invalid("native bridge response exceeds size bound"));
        }
        Ok(fs::read(output)?)
    }
}

fn rust_operation(request: &Request) -> Output {
    match request {
        Request::Pq2(input) => {
            let tensor = Tensor {
                name: "upstream-pq2-weights".to_owned(),
                kind: 142,
                dimensions: vec![input.k, input.m],
                data: Arc::from(input.packed.clone()),
            };
            // The runtime CPU API takes one activation column per call.
            // Preserve all upstream N columns and the original output order.
            let mut values = Vec::with_capacity(input.m * input.n);
            for activation in input.activations.chunks_exact(input.k) {
                match matvec(&tensor, activation) {
                    Ok(column) => values.extend(column),
                    Err(error) => return Output::Pq2(Err(error.to_string())),
                }
            }
            Output::Pq2(Ok(values))
        }
        Request::Ptq1(blocks) => Output::Ptq1(
            blocks
                .iter()
                .map(|packed| {
                    row(
                        &Tensor {
                            name: "upstream-ptq1-block".to_owned(),
                            kind: 143,
                            dimensions: vec![128],
                            data: Arc::from(*packed),
                        },
                        0,
                    )
                    .map_err(|error| error.to_string())
                })
                .collect(),
        ),
    }
}

fn push_u32(bytes: &mut Vec<u8>, value: usize) -> Result<()> {
    bytes.extend(u32::try_from(value)?.to_le_bytes());
    Ok(())
}

fn encode_requests(requests: &[Request]) -> Result<Vec<u8>> {
    let mut bytes = b"BSPREQ1\n".to_vec();
    push_u32(&mut bytes, requests.len())?;
    for request in requests {
        match request {
            Request::Pq2(input) => {
                for value in [
                    1,
                    input.k,
                    input.m,
                    input.n,
                    input.packed.len(),
                    input.activations.len(),
                ] {
                    push_u32(&mut bytes, value)?;
                }
                bytes.extend(&input.packed);
                for value in &input.activations {
                    bytes.extend(value.to_le_bytes());
                }
            }
            Request::Ptq1(blocks) => {
                for value in [2, 128, blocks.len(), 0, blocks.len() * 28, 0] {
                    push_u32(&mut bytes, value)?;
                }
                for block in blocks {
                    bytes.extend(block);
                }
            }
        }
    }
    Ok(bytes)
}

fn decode_response(bytes: &[u8], requests: &[Request]) -> Result<Vec<Output>> {
    let mut fixture = Fixture::new(bytes, *b"BSPRES1\n")?;
    if usize::try_from(fixture.u32()?)? != requests.len() {
        return Err(invalid("native bridge response request count mismatch"));
    }
    let mut outputs = Vec::with_capacity(requests.len());
    for request in requests {
        let expected = match request {
            Request::Pq2(input) => input.m * input.n,
            Request::Ptq1(blocks) => blocks.len() * 128,
        };
        if usize::try_from(fixture.u32()?)? != expected {
            return Err(invalid("native bridge response element count mismatch"));
        }
        let values = fixture.floats(expected)?;
        outputs.push(match request {
            Request::Pq2(_) => Output::Pq2(Ok(values)),
            Request::Ptq1(_) => Output::Ptq1(
                values
                    .as_chunks::<128>()
                    .0
                    .iter()
                    .map(|block| Ok(block.to_vec()))
                    .collect(),
            ),
        });
    }
    fixture.finish()?;
    Ok(outputs)
}

fn run_bridge(
    bridge: &Path,
    directory: &Path,
    input: &Path,
    output: &Path,
    timeout: Duration,
) -> Result<()> {
    let stdout_path = directory.join("stdout.log");
    let stderr_path = directory.join("stderr.log");
    let stdout = File::create(&stdout_path)?;
    let stderr = File::create(&stderr_path)?;
    let mut child = Command::new(bridge)
        .args([
            "--input".as_ref(),
            input.as_os_str(),
            "--output".as_ref(),
            output.as_os_str(),
        ])
        .current_dir(directory)
        .stdin(Stdio::null())
        .stdout(stdout)
        .stderr(stderr)
        .spawn()?;
    let started = Instant::now();
    let mut poll_interval = Duration::from_millis(1);
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                if exceeded_limits(started, timeout, &stdout_path, &stderr_path, output) {
                    return Err(invalid(format!(
                        "native bridge exceeded time/output bound; evidence: {}",
                        directory.display()
                    )));
                }
                if !status.success() {
                    return Err(invalid(format!(
                        "native bridge exited {status}; evidence: {}",
                        directory.display()
                    )));
                }
                return Ok(());
            }
            Ok(None) => {}
            Err(error) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(error.into());
            }
        }
        if exceeded_limits(started, timeout, &stdout_path, &stderr_path, output) {
            let _ = child.kill();
            let _ = child.wait();
            return Err(invalid(format!(
                "native bridge exceeded time/output bound; evidence: {}",
                directory.display()
            )));
        }
        // Short production calls should not each pay a fixed 20 ms polling
        // floor. Back off for long operations while retaining the same limits.
        std::thread::sleep(poll_interval);
        poll_interval = (poll_interval * 2).min(Duration::from_millis(20));
    }
}

fn exceeded_limits(
    started: Instant,
    timeout: Duration,
    stdout: &Path,
    stderr: &Path,
    output: &Path,
) -> bool {
    started.elapsed() > timeout
        || [stdout, stderr]
            .iter()
            .any(|path| fs::metadata(path).is_ok_and(|metadata| metadata.len() > MAX_LOG_BYTES))
        || fs::metadata(output).is_ok_and(|metadata| metadata.len() > MAX_RESPONSE_BYTES)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_wrong_bridge_counts_nonfinite_and_trailing_values() {
        let requests = [Request::Ptq1(vec![[0; 28]])];
        let mut response = b"BSPRES1\n".to_vec();
        response.extend(0_u32.to_le_bytes());
        assert!(decode_response(&response, &requests).is_err());
        response[8..].copy_from_slice(&1_u32.to_le_bytes());
        response.extend(127_u32.to_le_bytes());
        assert!(decode_response(&response, &requests).is_err());
        response[12..].copy_from_slice(&128_u32.to_le_bytes());
        response.extend(vec![0; 128 * 4]);
        assert!(decode_response(&response, &requests).is_ok());
        response.extend(0_u32.to_le_bytes());
        assert!(decode_response(&response, &requests).is_err());
        response.truncate(16 + 128 * 4);
        response[16..20].copy_from_slice(&f32::INFINITY.to_le_bytes());
        assert!(decode_response(&response, &requests).is_err());
    }
}
