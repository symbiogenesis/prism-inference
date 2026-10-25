//! Bounded safe process transport for the independent upstream Python implementation.

use std::ffi::OsString;
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::time::{Duration, Instant};

use super::engine::Engine;
use super::{Result, invalid};

const TIMEOUT: Duration = Duration::from_secs(30);
const MAX_STDOUT: u64 = 16 * 1024 * 1024;
const MAX_STDERR: u64 = 1024 * 1024;

/// Original optional Python branch decision, distinct from a target-engine result.
pub(super) enum Availability {
    Ready(PythonReference),
    Skipped(&'static str),
}

/// Absolute source and evidence paths for the independent production script.
pub(super) struct PythonReference {
    directory: PathBuf,
    script: PathBuf,
}

/// Actual process status and text-mode stdout, without any expected result.
pub(super) struct PythonOutput {
    pub(super) success: bool,
    pub(super) grammar: String,
}

struct Captured {
    status: ExitStatus,
    stdout: Vec<u8>,
}

fn exceeded(elapsed: Duration, stdout: u64, stderr: u64) -> bool {
    elapsed > TIMEOUT || stdout > MAX_STDOUT || stderr > MAX_STDERR
}

fn limits(started: Instant, stdout: &Path, stderr: &Path) -> Result<bool> {
    Ok(exceeded(
        started.elapsed(),
        fs::metadata(stdout)?.len(),
        fs::metadata(stderr)?.len(),
    ))
}

fn reap(child: &mut Child) {
    let _ = child.kill();
    let _ = child.wait();
}

fn capture(directory: &Path, stem: &str, arguments: &[OsString]) -> Result<Option<Captured>> {
    if !directory.is_absolute() {
        return Err(invalid("Python process working directory must be absolute"));
    }
    let stdout = directory.join(format!("{stem}.stdout"));
    let stderr = directory.join(format!("{stem}.stderr"));
    let mut child = match Command::new("python")
        .args(arguments)
        .current_dir(directory)
        .stdin(Stdio::null())
        .stdout(Stdio::from(File::create(&stdout)?))
        .stderr(Stdio::from(File::create(&stderr)?))
        .spawn()
    {
        Ok(child) => child,
        Err(error) => {
            fs::write(&stderr, format!("could not start Python: {error}"))?;
            return Ok(None);
        }
    };
    let started = Instant::now();
    loop {
        let status = match child.try_wait() {
            Ok(status) => status,
            Err(error) => {
                reap(&mut child);
                return Err(error.into());
            }
        };
        match limits(started, &stdout, &stderr) {
            Ok(false) => {}
            Ok(true) => {
                reap(&mut child);
                return Err(invalid(format!(
                    "Python reference exceeded time/output bound; evidence: {}",
                    directory.display()
                )));
            }
            Err(error) => {
                reap(&mut child);
                return Err(error);
            }
        }
        if let Some(status) = status {
            let _ = child.wait();
            return Ok(Some(Captured {
                status,
                stdout: fs::read(&stdout)?,
            }));
        }
        std::thread::sleep(Duration::from_millis(20));
    }
}

impl PythonReference {
    /// Preserves the source's emulator skip, explicit availability hint and >=3.8 probe.
    pub(super) fn prepare(engine: &Engine) -> Result<Availability> {
        if std::env::var_os("LLAMA_SKIP_TESTS_SLOW_ON_EMULATOR").is_some() {
            return Ok(Availability::Skipped(
                "LLAMA_SKIP_TESTS_SLOW_ON_EMULATOR is present",
            ));
        }
        let directory = engine.evidence_path("json-schema-python")?;
        if std::env::var_os("LLAMA_PYTHON_AVAILABLE").is_none() {
            let args = [
                OsString::from("-c"),
                OsString::from(
                    "import sys; exit(1) if sys.version_info < (3, 8) else print('Python version is sufficient')",
                ),
            ];
            if !capture(&directory, "availability", &args)?
                .is_some_and(|output| output.status.success())
            {
                return Ok(Availability::Skipped(
                    "Python>=3.8 probe was unavailable or unsuccessful",
                ));
            }
        }
        let script = engine.source_path("examples/json_schema_to_grammar.py")?;
        if !script.is_absolute() || !script.is_file() {
            return Err(invalid(
                "pinned production Python schema converter is missing",
            ));
        }
        Ok(Availability::Ready(Self { directory, script }))
    }

    /// Runs one supplied schema through the unmodified production Python CLI.
    pub(super) fn convert(&self, index: usize, schema: &str) -> Result<PythonOutput> {
        let stem = format!("case-{index:03}");
        let input = self.directory.join(format!("{stem}.json"));
        fs::write(&input, schema)?;
        let args = [
            self.script.as_os_str().to_os_string(),
            input.into_os_string(),
        ];
        let captured = capture(&self.directory, &stem, &args)?
            .ok_or_else(|| invalid("Python became unavailable after branch selection"))?;
        // C++ read() uses text-mode ifstream on Windows, which folds CRLF.
        let grammar = String::from_utf8(captured.stdout)?.replace("\r\n", "\n");
        Ok(PythonOutput {
            success: captured.status.success(),
            grammar,
        })
    }
}

/// Synthetic boundary predicate used by the focused fixture guard.
#[cfg(test)]
pub(super) fn synthetic_limits(elapsed: Duration, stdout: u64, stderr: u64) -> bool {
    exceeded(elapsed, stdout, stderr)
}
