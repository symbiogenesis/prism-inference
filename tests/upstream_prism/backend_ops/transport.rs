// SPDX-License-Identifier: MIT
//! Bounded continuation transport. Every action batch and continuation decision is Rust-owned.

use super::super::engine::Engine;
use super::super::{Result, invalid};
use serde_json::{Value, json};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::time::{Duration, Instant};

const MAX_OBSERVATION: u64 = 16 * 1024 * 1024;
const MAX_ACTION_FILE: usize = 128 * 1024 * 1024;
const MAX_ACTIONS: usize = 524_288;
pub(super) const MAX_FILE_BYTES: u64 = 4 * 1024 * 1024 * 1024;

type WorkerResult = Result<Option<Value>>;

#[path = "pipe.rs"]
mod pipe;
pub(super) use pipe::Partition;

pub(super) enum Session {
    File(FileSession),
    Pipe(pipe::Case),
    Rust(super::rust::GraphSession),
}

impl Session {
    pub(super) fn exchange(&mut self, actions: impl AsRef<[Value]>) -> Result<Vec<Value>> {
        let actions = actions.as_ref();
        match self {
            Self::File(session) => session.exchange(actions.to_vec()),
            Self::Pipe(session) => session.exchange(actions),
            Self::Rust(session) => session.exchange(actions),
        }
    }

    fn directory(&self) -> &Path {
        match self {
            Self::File(session) => &session.directory,
            Self::Pipe(session) => &session.directory,
            Self::Rust(session) => session.directory(),
        }
    }

    pub(super) fn file(&self, name: &str) -> Result<PathBuf> {
        let mut components = Path::new(name).components();
        if !matches!(components.next(), Some(Component::Normal(_))) || components.next().is_some() {
            return Err(invalid(
                "backend artifact name must be one normal path component",
            ));
        }
        Ok(self.directory().join(name))
    }

    pub(super) fn checked_file(&self, path: &Path) -> Result<PathBuf> {
        if !path.is_absolute() {
            return Err(invalid("backend observation file must be absolute"));
        }
        let path = fs::canonicalize(path)?;
        if !path.starts_with(self.directory()) || !path.is_file() {
            return Err(invalid(
                "backend observation file is outside its owned session directory",
            ));
        }
        Ok(path)
    }
}

pub(super) struct FileSession {
    directory: PathBuf,
    receiver: Receiver<WorkerResult>,
    completed: Option<WorkerResult>,
    output: PathBuf,
    next: PathBuf,
    sequence: usize,
    total_actions: usize,
    stopped: bool,
    timeout: Duration,
}

fn read_observation(path: &Path) -> Result<Value> {
    let metadata = fs::metadata(path)?;
    if !metadata.is_file() || metadata.len() > MAX_OBSERVATION {
        return Err(invalid("backend checkpoint is not a bounded regular file"));
    }
    let mut bytes = Vec::new();
    File::open(path)?
        .take(MAX_OBSERVATION + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() as u64 > MAX_OBSERVATION {
        return Err(invalid("backend checkpoint grew past its output bound"));
    }
    Ok(serde_json::from_slice(&bytes)?)
}

fn publish_actions(path: &Path, actions: &[Value]) -> Result<()> {
    let bytes = serde_json::to_vec(actions)?;
    if bytes.len() > MAX_ACTION_FILE {
        return Err(invalid("backend action file exceeds 128 MiB"));
    }
    let temporary = path.with_extension("pending");
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)?;
    file.write_all(&bytes)?;
    file.sync_all()?;
    drop(file);
    if path.try_exists()? {
        return Err(invalid("backend continuation input is not fresh"));
    }
    fs::rename(&temporary, path)?;
    Ok(())
}

fn checkpoint(directory: &Path, sequence: usize, timeout: Duration) -> (PathBuf, PathBuf, Value) {
    let output = directory.join(format!("checkpoint-{sequence:06}.json"));
    let next = directory.join(format!("actions-{:06}.json", sequence + 1));
    let action = json!({
        "Op":"checkpoint",
        "OutputFile":output,
        "NextFile":next,
        "TimeoutMilliseconds":timeout.as_millis().min(3_600_000)
    });
    (output, next, action)
}

impl FileSession {
    fn poll_worker(&mut self) -> Result<()> {
        if self.completed.is_some() {
            return Ok(());
        }
        match self.receiver.try_recv() {
            Ok(result) => self.completed = Some(result),
            Err(TryRecvError::Empty) => {}
            Err(TryRecvError::Disconnected) => {
                return Err(invalid("backend continuation worker disconnected"));
            }
        }
        Ok(())
    }

    fn wait_checkpoint(&mut self) -> Result<Option<Vec<Value>>> {
        let deadline = Instant::now() + self.timeout + Duration::from_secs(5);
        loop {
            if self.output.try_exists()? {
                let value = read_observation(&self.output)?;
                let object = value
                    .as_object()
                    .ok_or_else(|| invalid("backend checkpoint must be an object"))?;
                if object.len() != 1 || !object.contains_key("Trace") {
                    return Err(invalid("backend checkpoint fields differ"));
                }
                let mut trace = value["Trace"]
                    .as_array()
                    .ok_or_else(|| invalid("backend checkpoint Trace must be an array"))?
                    .clone();
                let acknowledgement = trace
                    .pop()
                    .ok_or_else(|| invalid("backend checkpoint omitted its acknowledgement"))?;
                if acknowledgement
                    .as_object()
                    .is_none_or(|value| !value.is_empty())
                {
                    return Err(invalid("backend checkpoint acknowledgement differs"));
                }
                return Ok(Some(trace));
            }
            self.poll_worker()?;
            if let Some(result) = self.completed.take() {
                match result? {
                    None => return Ok(None),
                    Some(_) => {
                        return Err(invalid(
                            "backend process completed before its requested checkpoint",
                        ));
                    }
                }
            }
            if Instant::now() >= deadline {
                return Err(invalid("backend checkpoint exceeded its bounded wait"));
            }
            std::thread::sleep(Duration::from_millis(2));
        }
    }

    pub(super) fn exchange(&mut self, mut actions: Vec<Value>) -> Result<Vec<Value>> {
        let supplied = actions.len();
        self.total_actions = self
            .total_actions
            .checked_add(supplied + 1)
            .ok_or_else(|| invalid("backend action extent overflow"))?;
        if supplied == 0 || self.total_actions > MAX_ACTIONS {
            return Err(invalid("backend explicit action extent exceeds its bound"));
        }
        self.sequence += 1;
        let (output, next, action) = checkpoint(&self.directory, self.sequence, self.timeout);
        actions.push(action);
        publish_actions(&self.next, &actions)?;
        self.output = output;
        self.next = next;
        let trace = self
            .wait_checkpoint()?
            .ok_or_else(|| invalid("native production API disappeared during a session"))?;
        if trace.len() != supplied {
            return Err(invalid("backend action/observation count differs"));
        }
        Ok(trace)
    }

    fn request_stop(&mut self) -> Result<()> {
        if !self.stopped {
            self.poll_worker()?;
            if self.completed.is_none() && !self.next.try_exists()? {
                publish_actions(&self.next, &[])?;
            }
            self.stopped = true;
        }
        Ok(())
    }

    fn finish(&mut self) -> Result<()> {
        self.request_stop()?;
        let result = if let Some(result) = self.completed.take() {
            result
        } else {
            self.receiver
                .recv_timeout(Duration::from_secs(95))
                .map_err(|_| invalid("backend worker did not finish after its stop request"))?
        };
        let Some(value) = result? else {
            return Err(invalid(
                "backend production API disappeared before completion",
            ));
        };
        let object = value
            .as_object()
            .ok_or_else(|| invalid("backend response must be an object"))?;
        let results = value["Results"]
            .as_array()
            .ok_or_else(|| invalid("backend response Results must be an array"))?;
        if object.len() != 2
            || value["SchemaVersion"] != 1
            || results.len() != 1
            || results[0].as_object().is_none_or(|value| value.len() != 1)
            || results[0]["Trace"] != json!([])
        {
            return Err(invalid("backend continuation completion framing differs"));
        }
        Ok(())
    }
}

/// The controller remains on the calling thread, so all source storage can use safe Rc cells.
pub(super) fn with_session<T>(
    engine: &Engine,
    directory: &Path,
    controller: impl FnOnce(&mut Session) -> Result<T>,
) -> Result<Option<T>> {
    if !directory.is_absolute() || !directory.is_dir() {
        return Err(invalid(
            "backend session directory must exist and be absolute",
        ));
    }
    let timeout = if std::env::var("GITHUB_ACTIONS").as_deref() == Ok("true")
        && std::env::var("GITHUB_REF").as_deref() == Ok("refs/heads/main")
    {
        Duration::from_secs(3600)
    } else {
        Duration::from_secs(60)
    };
    let process_timeout = if timeout > Duration::from_secs(60) {
        timeout
    } else {
        Duration::from_secs(90)
    };
    let (output, next, action) = checkpoint(directory, 0, timeout);
    let request = json!({
        "SchemaVersion":1,
        "Operations":[{
            "Kind":"backend-graph",
            "Directory":directory,
            "MaxFileBytes":MAX_FILE_BYTES,
            "Actions":[action]
        }]
    });
    std::thread::scope(|scope| {
        let (sender, receiver) = mpsc::channel();
        let worker = scope.spawn(move || {
            let outcome =
                engine.execute_common_with_timeout("backend-session", &request, process_timeout);
            let _ = sender.send(outcome);
        });
        let mut session = Session::File(FileSession {
            directory: fs::canonicalize(directory)?,
            receiver,
            completed: None,
            output,
            next,
            sequence: 0,
            total_actions: 1,
            stopped: false,
            timeout,
        });
        let Session::File(file) = &mut session else {
            return Err(invalid("backend file session differs"));
        };
        let ready = file.wait_checkpoint();
        let result = match ready {
            Ok(Some(trace)) if trace.is_empty() => controller(&mut session),
            Ok(None) => {
                worker
                    .join()
                    .map_err(|_| invalid("backend worker panicked"))?;
                return Ok(None);
            }
            Ok(Some(_)) => Err(invalid("backend initial checkpoint Trace is not empty")),
            Err(error) => Err(error),
        };
        let Session::File(file) = &mut session else {
            return Err(invalid("backend file session differs"));
        };
        let completion = file.finish();
        worker
            .join()
            .map_err(|_| invalid("backend worker panicked"))?;
        let result = result?;
        completion?;
        Ok(Some(result))
    })
}
