//! Owned native version and codec children using the existing audited boundary.

use std::ffi::OsString;
use std::fs::{self, OpenOptions};
use std::io::Write as _;
use std::path::Path;
use std::time::{Duration, Instant};

use bonsai_windows_process::{OwnedProcess, ParentProcess, ProcessSpec};
use serde_json::{Value, json};

const MEMORY_LIMIT: usize = 1024 * 1024 * 1024;
const OUTPUT_LIMIT: u64 = 16 * 1024 * 1024;

fn child(spec: &ProcessSpec, parent: &ParentProcess) -> Result<usize, String> {
    parent.ensure_alive().map_err(|error| error.to_string())?;
    let process = OwnedProcess::start(spec).map_err(|error| error.to_string())?;
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        parent.ensure_alive().map_err(|error| error.to_string())?;
        for path in [&spec.stdout, &spec.stderr] {
            if fs::metadata(path).map_err(|error| error.to_string())?.len() > OUTPUT_LIMIT {
                return Err("Native codec child exceeded its output limit.".into());
            }
        }
        if let Some(code) = process
            .wait_for_exit(Duration::from_millis(100))
            .map_err(|error| error.to_string())?
        {
            if code != 0 {
                return Err(format!("Native codec child failed with exit code {code}."));
            }
            return process
                .peak_job_memory_bytes()
                .map_err(|error| error.to_string());
        }
        if Instant::now() >= deadline {
            process
                .stop(Duration::from_secs(10))
                .map_err(|error| error.to_string())?;
            return Err("Native codec child exceeded its 30-second deadline.".into());
        }
    }
}

fn matches_commit(text: &str) -> bool {
    text.match_indices("commit ").any(|(start, _)| {
        let boundary = text[..start]
            .chars()
            .next_back()
            .is_none_or(|character| !character.is_ascii_alphanumeric() && character != '_');
        let Some((reported, _)) = text[start + 7..].split_once(')') else {
            return false;
        };
        boundary
            && (7..=40).contains(&reported.len())
            && reported.bytes().all(|byte| byte.is_ascii_hexdigit())
            && super::RUNTIME_COMMIT.starts_with(reported)
    })
}

#[expect(
    clippy::redundant_pub_crate,
    reason = "Binary implementation modules expose only to their parent."
)]
pub(super) fn run(arguments: &[OsString]) -> Result<Value, String> {
    if arguments.len() != 8
        || arguments[0] != "--runtime-directory"
        || arguments[2] != "--result"
        || arguments[4] != "--parent-process-id"
        || arguments[6] != "--parent-creation-time-file-time"
    {
        return Err("Usage: bonsai-codec-probe --runtime-directory ABSOLUTE_PINNED_DIRECTORY --result ABSOLUTE_JSON_PATH --parent-process-id PID --parent-creation-time-file-time FILETIME".into());
    }
    let runtime = Path::new(&arguments[1]);
    let result = Path::new(&arguments[3]);
    if !runtime.is_absolute()
        || runtime
            .file_name()
            .is_none_or(|name| name != super::RUNTIME_COMMIT)
        || !result.is_absolute()
        || result.exists()
    {
        return Err(
            "Use a pinned absolute runtime directory and a fresh absolute result file.".into(),
        );
    }
    let parse = |index: usize| {
        arguments[index]
            .to_str()
            .ok_or("Parent identity must be decimal.")?
            .parse::<u64>()
            .map_err(|error| error.to_string())
    };
    let parent_id = u32::try_from(parse(5)?).map_err(|error| error.to_string())?;
    let parent = ParentProcess::open(parent_id, parse(7)?).map_err(|error| error.to_string())?;
    let evidence = result.parent().ok_or("Missing result parent directory.")?;
    let mut spec = ProcessSpec {
        executable: runtime.join("llama-server.exe"),
        arguments: vec![OsString::from("--version")],
        working_directory: evidence.to_path_buf(),
        stdout: evidence.join("version.stdout"),
        stderr: evidence.join("version.stderr"),
        memory_limit_bytes: MEMORY_LIMIT,
        active_process_limit: 1,
    };
    let version_peak = child(&spec, &parent)?;
    let version = fs::read_to_string(&spec.stdout).map_err(|error| error.to_string())?
        + &fs::read_to_string(&spec.stderr).map_err(|error| error.to_string())?;
    if !matches_commit(&version) {
        return Err("Native runtime version does not identify the pinned commit.".into());
    }
    spec.executable = std::env::current_exe().map_err(|error| error.to_string())?;
    spec.arguments = vec![
        OsString::from("--native-child"),
        OsString::from("--runtime-directory"),
        runtime.as_os_str().to_os_string(),
        OsString::from("--result"),
        result.as_os_str().to_os_string(),
    ];
    spec.stdout = evidence.join("codec.stdout");
    spec.stderr = evidence.join("codec.stderr");
    let codec_peak = child(&spec, &parent)?;
    let receipt = json!({
        "Status":"owned-native-codec-complete", "NativeRuntimeCommit":super::RUNTIME_COMMIT,
        "RuntimeVersion":version.trim(), "MemoryLimitBytes":MEMORY_LIMIT,
        "ActiveProcessLimit":1, "TimeoutSeconds":30, "ModelLoaded":false,
        "VersionPeakJobMemoryBytes":version_peak, "CodecPeakJobMemoryBytes":codec_peak,
    });
    let mut output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(evidence.join("supervision.json"))
        .map_err(|error| error.to_string())?;
    output
        .write_all(&serde_json::to_vec_pretty(&receipt).map_err(|error| error.to_string())?)
        .map_err(|error| error.to_string())?;
    Ok(receipt)
}
