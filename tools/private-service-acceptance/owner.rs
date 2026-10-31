//! Prism-owned capture using the exact reviewed Runner launcher as process owner.
//!
//! Hosted preparation copies this example into the pinned Runner checkout. No
//! dependency or launch policy is changed, and the device never builds source.
#![forbid(unsafe_code)]

use bonsai_artifacts::VerifiedFile;
use bonsai_broker::{launcher::LaunchConfig, runtime::RunExit};
use serde_json::{Value, json};
use std::{
    cell::Cell,
    env,
    ffi::OsString,
    fs::{File, OpenOptions},
    future::pending,
    io::{self, Seek},
    net::TcpListener,
    path::{Path, PathBuf},
    process::{Child, Command, ExitCode, Stdio},
    time::{Duration, Instant},
};
use tokio::time::{sleep, timeout};

const RUNNER_REVISION: &str = "1914330f8a31769bfe8368ca9728fe7a418ed84e";
const RELEASE: &str = include_str!("../../../config/prism-inference.lock.json");
const CLEANUP: Duration = Duration::from_secs(20);

fn invalid() -> io::Error {
    io::Error::other("private service owner capture failed")
}

fn require(condition: bool) -> io::Result<()> {
    if condition { Ok(()) } else { Err(invalid()) }
}

fn seconds(value: &str) -> io::Result<u64> {
    require(!value.is_empty() && value.bytes().all(|b| b.is_ascii_digit()))?;
    let seconds = value.parse().map_err(|_| invalid())?;
    require((1..=240).contains(&seconds))?;
    Ok(seconds)
}

fn remaining(deadline: Instant) -> io::Result<Duration> {
    deadline
        .checked_duration_since(Instant::now())
        .filter(|value| !value.is_zero())
        .ok_or_else(invalid)
}

struct Receipt {
    file: File,
    value: Value,
}

impl Receipt {
    fn save(&mut self) -> io::Result<()> {
        self.file.rewind()?;
        serde_json::to_writer_pretty(&mut self.file, &self.value).map_err(|_| invalid())?;
        let length = self.file.stream_position()?;
        self.file.set_len(length)?;
        self.file.sync_all()
    }
}

/// A child may never survive a failed owner future or a failed guard.
struct Checker(Child);

impl Drop for Checker {
    fn drop(&mut self) {
        if !matches!(self.0.try_wait(), Ok(Some(_))) {
            let _result = self.0.kill();
        }
        let _result = self.0.wait();
    }
}

fn launch(data: &Path, evidence: &Path, port: u16) -> io::Result<LaunchConfig> {
    let mut arguments: Vec<OsString> = [
        "--launch-development",
        "--require-native-sandbox",
        "--native-engine",
        "--native-row-workers",
        "4",
        "--use-abliterate-lora",
        "--profile",
        "swift-pq2-assessment",
        "--frontend-port",
        "0",
        "--backend-port",
    ]
    .into_iter()
    .map(OsString::from)
    .collect();
    arguments.push(port.to_string().into());
    arguments.push("--data-directory".into());
    arguments.push(data.as_os_str().to_owned());
    arguments.push("--evidence-parent".into());
    arguments.push(evidence.as_os_str().to_owned());
    LaunchConfig::parse(arguments).map_err(|_| invalid())
}

fn guard_release(data: &Path, receipt: &mut Receipt) -> io::Result<Vec<VerifiedFile>> {
    let lock: Value = serde_json::from_str(RELEASE).map_err(|_| invalid())?;
    let revision = lock["commit"].as_str().ok_or_else(invalid)?;
    let target = lock["target"].as_str().ok_or_else(invalid)?;
    let mut guards = Vec::new();
    let mut entries = Vec::new();
    let files = lock["files"].as_array().ok_or_else(invalid)?;
    for file in files {
        let name = file["name"].as_str().ok_or_else(invalid)?;
        entries.push((
            data.join("inference")
                .join(revision)
                .join(target)
                .join(name),
            file,
        ));
    }
    entries.push((
        data.join("downloads")
            .join(lock["asset"]["filename"].as_str().ok_or_else(invalid)?),
        &lock["asset"],
    ));
    for (path, file) in entries {
        guards.push(
            VerifiedFile::verify(
                &path,
                file["bytes"].as_u64().ok_or_else(invalid)?,
                file["sha256"].as_str().ok_or_else(invalid)?,
            )
            .map_err(|_| invalid())?,
        );
    }
    let identity = |name: &str| -> io::Result<Value> {
        files
            .iter()
            .find(|file| file["name"] == name)
            .cloned()
            .ok_or_else(invalid)
    };
    receipt.value["testedRuntime"] = json!({"revision":revision,"version":lock["version"],
        "archiveSha256":lock["asset"]["sha256"],
        "serverSha256":identity("prism-inference-server.exe")?["sha256"],
        "workerSha256":identity("bonsai-swift-worker.exe")?["sha256"]});
    receipt.value["releasedFilesAndArchiveVerified"] = json!(true);
    receipt.save()?;
    Ok(guards)
}

#[allow(
    clippy::future_not_send,
    reason = "the reviewed launcher and shutdown are polled on one current-thread runtime"
)]
async fn capture(
    data: &Path,
    evidence: &Path,
    checker: &Path,
    deadline: Instant,
    receipt: &mut Receipt,
) -> io::Result<()> {
    let _release = guard_release(data, receipt)?;
    let reservation = TcpListener::bind("127.0.0.1:0")?;
    let backend_port = reservation.local_addr()?.port();
    drop(reservation);
    let owner = timeout(
        remaining(deadline)?
            .checked_sub(CLEANUP)
            .ok_or_else(invalid)?,
        launch(data, evidence, backend_port)?.start(pending::<io::Result<()>>()),
    )
    .await
    .map_err(|_| invalid())?
    .map_err(|_| invalid())?;
    receipt.value["authenticatedOwnedReadiness"] = json!(true);
    receipt.save()?;
    let frontend_port = owner.port();
    let key = owner.evidence_directory().join("backend.key");
    let check_seconds = remaining(deadline)?
        .checked_sub(CLEANUP)
        .ok_or_else(invalid)?
        .as_secs()
        .min(240);
    require(check_seconds > 0)?;
    let mut command = Command::new(checker);
    command
        .arg(backend_port.to_string())
        .arg(key)
        .arg(evidence.join("service.json"))
        .arg(check_seconds.to_string())
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    for name in [
        "ACTIONS_RUNTIME_TOKEN",
        "ACTIONS_ID_TOKEN_REQUEST_TOKEN",
        "GITHUB_TOKEN",
        "GH_TOKEN",
        "ACTIONS_RUNNER_INPUT_TOKEN",
    ] {
        command.env_remove(name);
    }
    let mut child = Checker(command.spawn()?);
    let check_deadline = Instant::now() + Duration::from_secs(check_seconds);
    let completed = Cell::new(false);
    let shutdown = async {
        loop {
            if let Some(status) = child.0.try_wait()? {
                completed.set(status.success());
                return Ok(());
            }
            if remaining(check_deadline).is_err() {
                return Ok(());
            }
            sleep(Duration::from_millis(25)).await;
        }
    };
    let result = owner.run(shutdown).await.map_err(|_| invalid())?;
    // LaunchedService::run returns success only after confirming the Job empty.
    receipt.value["ownedJobEmpty"] = json!(true);
    require(result.exit == RunExit::Shutdown)?;
    let _backend_released = TcpListener::bind(("127.0.0.1", backend_port))?;
    let _frontend_released = TcpListener::bind(("127.0.0.1", frontend_port))?;
    receipt.value["listenersReleased"] = json!(true);
    receipt.value["checkerSucceeded"] = json!(completed.get());
    receipt.save()?;
    require(completed.get())?;
    remaining(deadline).map(|_| ())
}

fn run() -> io::Result<()> {
    require(
        env::var("GITHUB_ACTIONS").as_deref() == Ok("true")
            && env::var("GITHUB_REF").as_deref() == Ok("refs/heads/main"),
    )?;
    require(env::consts::OS == "windows" && env::consts::ARCH == "aarch64")?;
    require(env::var("GITHUB_SHA").as_deref() == Ok(env!("PRISM_HARNESS_REVISION")))?;
    let arguments: Vec<_> = env::args_os().skip(1).collect();
    require(arguments.len() == 4)?;
    let data = PathBuf::from(&arguments[0]);
    let evidence = PathBuf::from(&arguments[1]);
    let checker = PathBuf::from(&arguments[2]);
    require(data.is_absolute() && evidence.is_absolute() && checker.is_absolute())?;
    let allowance = seconds(arguments[3].to_str().ok_or_else(invalid)?)?;
    let started = Instant::now();
    let deadline = started + Duration::from_secs(allowance);
    let file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(evidence.join("owner.json"))?;
    let mut receipt = Receipt {
        file,
        value: json!({"schemaVersion":1,"status":"pending",
        "harnessRevision":env!("PRISM_HARNESS_REVISION"),"runnerOwnerRevision":RUNNER_REVISION,
        "aggregateOwnerAllowanceSeconds":allowance,"elapsedMilliseconds":0,
        "profile":"original-pq2-approved-adapter-row4-reset-every-request",
        "rowWorkersRequested":4,"releasedFilesAndArchiveVerified":false,
        "authenticatedOwnedReadiness":false,"ownedJobEmpty":false,"listenersReleased":false,
        "checkerSucceeded":false,"separateJsonlWorkerExecuted":false,
        "activeGraphCancellationQualified":false,"completeStateRollbackQualified":false,
        "defaultPromotionQualified":false}),
    };
    receipt.save()?;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let result = runtime.block_on(capture(&data, &evidence, &checker, deadline, &mut receipt));
    receipt.value["elapsedMilliseconds"] =
        json!(u64::try_from(started.elapsed().as_millis()).map_err(|_| invalid())?);
    receipt.value["status"] = json!(if result.is_ok() { "passed" } else { "failed" });
    receipt.save()?;
    result
}

fn main() -> ExitCode {
    if run().is_ok() {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn owner_allowance_is_required_bounded_and_decimal() {
        for value in [
            "",
            "0",
            "241",
            "-1",
            "+1",
            " 1",
            "1.0",
            "99999999999999999999999",
        ] {
            assert!(seconds(value).is_err());
        }
        assert_eq!(seconds("1").ok(), Some(1));
        assert_eq!(seconds("240").ok(), Some(240));
    }

    #[test]
    fn owner_profile_keeps_required_sandbox_and_native_selection() -> io::Result<()> {
        let config = launch(Path::new(r"C:\Data"), Path::new(r"C:\Evidence"), 12000)?;
        assert!(config.requires_native_sandbox() && config.uses_native_engine());
        assert!(!config.uses_repacked_ptq1());
        Ok(())
    }
}
