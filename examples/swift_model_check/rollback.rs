//! Source-graph cancellation evidence; never private HTTP service qualification.

use super::{AdapterSelection, Representation, evidence_backend, load_graph, reject, valid_tokens};
use bonsai_prism_runtime::{
    Decoder, Result,
    graph::{Qwen35, StateComponent},
    model::swift,
    runtime::{Cancellation, Finish, Generation, Limits, Prefill, Runtime},
    sampling::Sampler,
};
use serde::Serialize;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    fs::{File, OpenOptions},
    io::{Read, Seek},
    path::Path,
};

const MAX_CAPTURE_BYTES: usize = 2 * 1024 * 1024 * 1024;
const CHECKPOINT_BYTES: usize = 256 * 1024 * 1024;

trait ObservedState: Decoder {
    fn visit(
        &self,
        observer: &mut impl FnMut(usize, StateComponent, &[f32]) -> Result<()>,
    ) -> Result<()>;
}

impl ObservedState for Qwen35 {
    fn visit(
        &self,
        observer: &mut impl FnMut(usize, StateComponent, &[f32]) -> Result<()>,
    ) -> Result<()> {
        self.visit_state(observer)
    }
}

// Returning a declared stop records the complete logits without evaluating or
// emitting another token. On exact live-prefix probes, neither graph state nor
// committed history changes. This is a diagnostic, not a production sampler.
#[derive(Clone, Default)]
struct CaptureStop(Vec<f32>);

impl Sampler for CaptureStop {
    fn sample(&mut self, logits: &[f32]) -> Result<u32> {
        if logits.is_empty() || logits.iter().any(|value| !value.is_finite()) {
            return Err(reject("rollback capture requires complete finite logits"));
        }
        self.0.clear();
        self.0.try_reserve_exact(logits.len())?;
        self.0.extend_from_slice(logits);
        Ok(0)
    }
}

fn observe_logits<D: Decoder>(runtime: &mut Runtime<D>, exact_live: bool) -> Result<Vec<f32>> {
    let tokens = runtime.tokens().to_vec();
    let mut sampler = CaptureStop::default();
    let completion = runtime.generate_sampled(
        &tokens,
        &Generation {
            max_tokens: 1,
            stop_tokens: vec![0],
        },
        Prefill::default(),
        &Cancellation::default(),
        &mut sampler,
        |_| Err(reject("capture stop must not emit a token")),
    )?;
    if completion.finish != Finish::Stop
        || completion.generated_tokens != 0
        || (exact_live && completion.reused_tokens != tokens.len())
        || runtime.tokens() != tokens
        || sampler.0.len() != runtime.decoder().vocab_size()
    {
        return Err(reject("logits probe must preserve the exact live prefix"));
    }
    Ok(sampler.0)
}

#[derive(Serialize, PartialEq, Eq, Debug)]
struct CaptureFile {
    file: String,
    values: usize,
    bytes: u64,
    sha256: String,
}

#[derive(Serialize)]
struct Capture {
    tokens: Vec<u32>,
    files: Vec<CaptureFile>,
}

fn recorded_file(path: &Path, values: usize) -> Result<CaptureFile> {
    let mut file = File::open(path)?;
    let bytes = file.metadata()?.len();
    let mut digest = Sha256::new();
    let mut buffer = [0; 16_384];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(CaptureFile {
        file: path
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| reject("invalid capture filename"))?
            .to_owned(),
        values,
        bytes,
        sha256: digest
            .finalize()
            .iter()
            .flat_map(|byte| [byte >> 4, byte & 15])
            .map(|nibble| char::from(b"0123456789abcdef"[usize::from(nibble)]))
            .collect(),
    })
}

fn capture<D: ObservedState>(
    runtime: &Runtime<D>,
    logits: &[f32],
    directory: &Path,
    remaining: &mut usize,
) -> Result<Capture> {
    if runtime.is_faulted() || logits.len() != runtime.decoder().vocab_size() {
        return Err(reject("cannot capture faulted or incomplete runtime state"));
    }
    std::fs::create_dir(directory)?;
    let mut files = Vec::new();
    let output = directory.join("logits.f32le");
    super::write_values(&output, logits, remaining)?;
    files.push(recorded_file(&output, logits.len())?);
    runtime.decoder().visit(&mut |block, component, values| {
        let label = match component {
            StateComponent::Key(token) => format!("key-{token}"),
            StateComponent::Value(token) => format!("value-{token}"),
            StateComponent::Convolution => "convolution".to_owned(),
            StateComponent::Recurrent => "recurrent".to_owned(),
        };
        let output = directory.join(format!("block-{block}-{label}.f32le"));
        super::write_values(&output, values, remaining)?;
        files.push(recorded_file(&output, values.len())?);
        Ok(())
    })?;
    let capture = Capture {
        tokens: runtime.tokens().to_vec(),
        files,
    };
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(directory.join("capture.json"))?;
    serde_json::to_writer_pretty(file, &capture)?;
    Ok(capture)
}

fn identical(expected: &Capture, actual: &Capture) -> Result<()> {
    if expected.tokens != actual.tokens || expected.files != actual.files {
        return Err(reject(
            "complete captured state, tokens or logits differ bit for bit",
        ));
    }
    Ok(())
}

struct Receipt {
    file: File,
    value: serde_json::Value,
}

impl Receipt {
    fn new(directory: &Path, identity: &serde_json::Value) -> Result<Self> {
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(directory.join("rollback.json"))?;
        let mut receipt = Self {
            file,
            value: json!({
                "schemaVersion":1,"status":"pending","identity":identity,"completedSteps":[],
                "scope":"source-graph-transactional-cancellation-and-fresh-continuation",
                "comparison":"all complete logical KV/convolution/recurrent values and logits as exact f32le bits, plus exact committed token IDs",
                "captureStopToken":0,"captureStopIsProductionSampler":false,
                "maximumSnapshotPayloadBytes":MAX_CAPTURE_BYTES,"rollbackCheckpointBudgetBytes":CHECKPOINT_BYTES,
                "privateHttpServiceQualified":false,"activeGraphInterruptionQualified":false,
                "productionSamplerQualified":false,"performanceQualified":false,"defaultPromotionQualified":false
            }),
        };
        receipt.save()?;
        Ok(receipt)
    }

    fn save(&mut self) -> Result<()> {
        self.file.rewind()?;
        serde_json::to_writer_pretty(&mut self.file, &self.value)?;
        let length = self.file.stream_position()?;
        self.file.set_len(length)?;
        self.file.sync_all()?;
        Ok(())
    }

    fn step(&mut self, step: &str) -> Result<()> {
        self.value["completedSteps"]
            .as_array_mut()
            .ok_or_else(|| reject("invalid capture receipt"))?
            .push(json!(step));
        self.save()
    }

    fn finish(&mut self, result: Result<()>) -> Result<()> {
        self.value["status"] = json!(if result.is_ok() { "passed" } else { "failed" });
        self.save()?;
        result
    }
}

#[allow(
    clippy::redundant_pub_crate,
    reason = "The source example's private helper is callable only by its parent dispatcher."
)]
pub(super) fn run(
    path: &Path,
    arguments: &[String],
    representation: Representation,
    adapter: Option<&AdapterSelection>,
    row_workers: usize,
) -> Result<()> {
    if arguments.len() != 3 {
        return Err(reject(
            "rollback requires output directory, backend and 1..16 token IDs",
        ));
    }
    let (evidence, backend) = evidence_backend(arguments)?;
    let prompt = arguments[2]
        .split(',')
        .map(str::parse::<u32>)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    if !valid_tokens(&prompt) || prompt.len() > 16 {
        return Err(reject("rollback requires 1..16 valid token IDs"));
    }
    let mut receipt = Receipt::new(
        evidence,
        &json!({
            "modelSha256":representation.model_sha256(),"representation":representation.name(),
            "adapterScaleBits":adapter.map(|selection| format!("{:08x}", selection.scale.to_bits())),
            "adapterSha256":adapter.map(|_| bonsai_prism_runtime::model::lora::SWIFT_ADAPTER_SHA256),
            "rowWorkers":row_workers,"backendRequested":arguments[1],"promptTokens":prompt
        }),
    )?;
    let result = (|| {
        let graph = load_graph(path, backend, representation, adapter, row_workers)?;
        receipt.step("exact-model-and-profile-loaded")?;
        exercise(graph, &prompt, evidence, &mut receipt)
    })();
    receipt.finish(result)
}

fn exercise<D: ObservedState>(
    decoder: D,
    prompt: &[u32],
    evidence: &Path,
    receipt: &mut Receipt,
) -> Result<()> {
    let mut runtime = Runtime::new(
        decoder,
        Limits {
            context_tokens: swift::CONTEXT_TOKENS,
            checkpoints: 0,
            checkpoint_bytes: CHECKPOINT_BYTES,
        },
    )?;
    let mut initial = CaptureStop::default();
    runtime.generate_sampled(
        prompt,
        &Generation {
            max_tokens: 1,
            stop_tokens: vec![0],
        },
        Prefill::default(),
        &Cancellation::default(),
        &mut initial,
        |_| Err(reject("capture stop emitted a token")),
    )?;
    let mut remaining = MAX_CAPTURE_BYTES;
    let before = capture(
        &runtime,
        &initial.0,
        &evidence.join("before"),
        &mut remaining,
    )?;
    receipt.step("complete-pre-cancellation-state-and-logits-captured")?;

    let cancelled_token = cancel_after_first(&mut runtime, prompt)?;
    receipt.value["cancelledProvisionalToken"] = json!(cancelled_token);
    receipt.step("extended-prompt-and-one-evaluated-token-cancelled")?;
    let logits = observe_logits(&mut runtime, true)?;
    let restored = capture(
        &runtime,
        &logits,
        &evidence.join("restored"),
        &mut remaining,
    )?;
    identical(&before, &restored)?;
    receipt.step("complete-state-tokens-and-logits-restored-before-any-reset")?;

    let resumed = continue_one(&mut runtime, prompt)?;
    let logits = observe_logits(&mut runtime, true)?;
    let continuation = capture(
        &runtime,
        &logits,
        &evidence.join("continuation"),
        &mut remaining,
    )?;
    receipt.step("post-rollback-one-token-continuation-captured")?;
    runtime.reset()?;
    let fresh = continue_one(&mut runtime, prompt)?;
    let logits = observe_logits(&mut runtime, true)?;
    let recomputed = capture(&runtime, &logits, &evidence.join("fresh"), &mut remaining)?;
    if resumed != fresh {
        return Err(reject(
            "rollback continuation differs from fresh recomputation",
        ));
    }
    identical(&continuation, &recomputed)?;
    receipt.value["continuationToken"] = json!(resumed);
    receipt.value["snapshotPayloadBytesWritten"] = json!(MAX_CAPTURE_BYTES - remaining);
    receipt.step("fresh-recompute-continuation-token-full-state-and-logits-match")
}

fn cancel_after_first<D: Decoder>(runtime: &mut Runtime<D>, prompt: &[u32]) -> Result<u32> {
    let mut extended = prompt.to_vec();
    extended.push(
        *prompt
            .last()
            .ok_or_else(|| reject("empty rollback prompt"))?,
    );
    let cancellation = Cancellation::default();
    let mut provisional = Vec::new();
    let result = runtime.generate(
        &extended,
        &Generation {
            max_tokens: 2,
            stop_tokens: Vec::new(),
        },
        &cancellation,
        |token| {
            provisional.push(token);
            cancellation.cancel();
            Ok(())
        },
    );
    let interrupted = result
        .as_ref()
        .err()
        .and_then(|error| error.downcast_ref::<std::io::Error>())
        .is_some_and(|error| error.kind() == std::io::ErrorKind::Interrupted);
    if !interrupted
        || !cancellation.is_cancelled()
        || provisional.len() != 1
        || runtime.is_faulted()
        || runtime.tokens() != prompt
    {
        return Err(reject(
            "expected one evaluated provisional token and successful cancellation rollback",
        ));
    }
    Ok(provisional[0])
}

fn continue_one<D: Decoder>(runtime: &mut Runtime<D>, prompt: &[u32]) -> Result<u32> {
    let mut output = Vec::new();
    let completion = runtime.generate(
        prompt,
        &Generation {
            max_tokens: 1,
            stop_tokens: Vec::new(),
        },
        &Cancellation::default(),
        |token| {
            output.push(token);
            Ok(())
        },
    )?;
    if completion.finish != Finish::Length || completion.generated_tokens != 1 || output.len() != 1
    {
        return Err(reject("expected exactly one greedy continuation token"));
    }
    Ok(output[0])
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Fake {
        tokens: Vec<u32>,
        recurrent: f32,
        incomplete_restore: bool,
        fail_extended: bool,
    }

    impl Decoder for Fake {
        type Checkpoint = (Vec<u32>, f32);
        fn identity(&self) -> &'static str {
            "rollback-capture-control"
        }
        fn vocab_size(&self) -> usize {
            4
        }
        fn checkpoint_bytes(&self) -> usize {
            256
        }
        fn reset(&mut self) -> Result<()> {
            self.tokens.clear();
            self.recurrent = -0.0;
            Ok(())
        }
        fn evaluate(&mut self, token: u32) -> Result<Vec<f32>> {
            if self.fail_extended && self.tokens.len() >= 2 {
                return Err(std::io::Error::other("injected pre-callback graph failure").into());
            }
            self.tokens.push(token);
            self.recurrent += 1.0 + f32::from(u16::try_from(token)?);
            Ok(vec![-0.0, self.recurrent, -1.0, 0.0])
        }
        fn checkpoint(&self) -> Result<Self::Checkpoint> {
            Ok((self.tokens.clone(), self.recurrent))
        }
        fn restore(&mut self, checkpoint: &Self::Checkpoint) -> Result<()> {
            self.tokens.clone_from(&checkpoint.0);
            if !self.incomplete_restore {
                self.recurrent = checkpoint.1;
            }
            Ok(())
        }
    }

    impl ObservedState for Fake {
        fn visit(
            &self,
            observer: &mut impl FnMut(usize, StateComponent, &[f32]) -> Result<()>,
        ) -> Result<()> {
            for (index, token) in self.tokens.iter().enumerate() {
                observer(
                    0,
                    StateComponent::Key(index),
                    &[f32::from(u16::try_from(*token)?), -0.0],
                )?;
                observer(
                    0,
                    StateComponent::Value(index),
                    &[0.0, f32::from(u16::try_from(*token)?)],
                )?;
            }
            observer(1, StateComponent::Convolution, &[0.0, -0.0])?;
            observer(1, StateComponent::Recurrent, &[self.recurrent])
        }
    }

    struct TestDirectory(std::path::PathBuf, std::path::PathBuf);
    impl TestDirectory {
        fn new() -> Result<Self> {
            let nonce = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_nanos();
            let path = std::env::temp_dir().join(format!(
                "Prism-Rollback-Capture-{}-{nonce}",
                std::process::id()
            ));
            std::fs::create_dir(&path)?;
            Ok(Self(
                path.canonicalize()?,
                std::env::temp_dir().canonicalize()?,
            ))
        }
    }
    impl Drop for TestDirectory {
        fn drop(&mut self) {
            if let Ok(resolved) = self.0.canonicalize()
                && resolved == self.0
                && resolved.parent() == Some(self.1.as_path())
            {
                let _ = std::fs::remove_dir_all(resolved);
            }
        }
    }

    #[test]
    fn complete_capture_checks_rollback_before_reset_and_fresh_continuation() -> Result<()> {
        let directory = TestDirectory::new()?;
        let mut receipt = Receipt::new(&directory.0, &json!({"fixture":true}))?;
        let decoder = Fake {
            tokens: Vec::new(),
            recurrent: 0.0,
            incomplete_restore: false,
            fail_extended: false,
        };
        let result = exercise(decoder, &[1, 2], &directory.0, &mut receipt);
        receipt.finish(result)?;
        let report: serde_json::Value =
            serde_json::from_reader(File::open(directory.0.join("rollback.json"))?)?;
        assert_eq!(report["status"], "passed");
        assert_eq!(report["cancelledProvisionalToken"], 1);
        assert_eq!(report["completedSteps"].as_array().map(Vec::len), Some(5));
        for flag in [
            "privateHttpServiceQualified",
            "activeGraphInterruptionQualified",
            "defaultPromotionQualified",
        ] {
            assert_eq!(report[flag], false);
        }
        let before: serde_json::Value =
            serde_json::from_reader(File::open(directory.0.join("before/capture.json"))?)?;
        assert_eq!(before["tokens"], json!([1, 2]));
        // Two KV key/value rows, convolution, recurrent, and the full logits.
        assert_eq!(before["files"].as_array().map(Vec::len), Some(7));
        assert_eq!(
            std::fs::read(directory.0.join("before/block-1-convolution.f32le"))?,
            [0, 0, 0, 0, 0, 0, 0, 128]
        );
        assert!(Receipt::new(&directory.0, &json!({})).is_err());
        Ok(())
    }

    #[test]
    fn incomplete_recurrent_restore_cannot_pass_after_an_explicit_reset() -> Result<()> {
        let directory = TestDirectory::new()?;
        let mut receipt = Receipt::new(&directory.0, &json!({"fixture":true}))?;
        let decoder = Fake {
            tokens: Vec::new(),
            recurrent: 0.0,
            incomplete_restore: true,
            fail_extended: false,
        };
        let result = exercise(decoder, &[1, 2], &directory.0, &mut receipt);
        assert!(receipt.finish(result).is_err());
        let report: serde_json::Value =
            serde_json::from_reader(File::open(directory.0.join("rollback.json"))?)?;
        assert_eq!(report["status"], "failed");
        assert_eq!(report["completedSteps"].as_array().map(Vec::len), Some(2));
        assert!(directory.0.join("restored/capture.json").is_file());
        assert!(!directory.0.join("continuation").exists());
        assert!(!directory.0.join("fresh").exists());
        Ok(())
    }

    #[test]
    fn unrelated_graph_failure_is_not_reported_as_cancellation() -> Result<()> {
        let directory = TestDirectory::new()?;
        let mut receipt = Receipt::new(&directory.0, &json!({"fixture":true}))?;
        let decoder = Fake {
            tokens: Vec::new(),
            recurrent: 0.0,
            incomplete_restore: false,
            fail_extended: true,
        };
        let result = exercise(decoder, &[1, 2], &directory.0, &mut receipt);
        assert!(receipt.finish(result).is_err());
        let report: serde_json::Value =
            serde_json::from_reader(File::open(directory.0.join("rollback.json"))?)?;
        assert_eq!(report["status"], "failed");
        assert_eq!(report["completedSteps"].as_array().map(Vec::len), Some(1));
        assert!(report.get("cancelledProvisionalToken").is_none());
        Ok(())
    }

    #[test]
    fn capture_budget_and_existing_directories_are_hard_failures() -> Result<()> {
        let directory = TestDirectory::new()?;
        let mut runtime = Runtime::new(
            Fake {
                tokens: Vec::new(),
                recurrent: 0.0,
                incomplete_restore: false,
                fail_extended: false,
            },
            Limits {
                context_tokens: 16,
                checkpoints: 0,
                checkpoint_bytes: 4096,
            },
        )?;
        runtime.generate(
            &[1],
            &Generation {
                max_tokens: 1,
                stop_tokens: Vec::new(),
            },
            &Cancellation::default(),
            |_| Ok(()),
        )?;
        let logits = observe_logits(&mut runtime, true)?;
        let output = directory.0.join("bounded");
        let mut remaining = 15;
        assert!(capture(&runtime, &logits, &output, &mut remaining).is_err());
        assert_eq!(remaining, 15);
        assert!(!output.join("logits.f32le").exists());
        remaining = MAX_CAPTURE_BYTES;
        assert!(capture(&runtime, &logits, &output, &mut remaining).is_err());
        assert_eq!(remaining, MAX_CAPTURE_BYTES);
        Ok(())
    }
}
