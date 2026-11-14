//! Windows Job supervision, temporary-index checkout and trusted snapshot checks.

use std::collections::{BTreeMap, BTreeSet};
use std::ffi::OsString;
use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use bonsai_windows_process::{OwnedProcess, ToolProcessSpec};

use super::{Result, Update};

const WORK_LIMIT: Duration = Duration::from_secs(270);
const OUTPUT_LIMIT: u64 = 4 * 1024 * 1024;
const REMOVED_GIT_ENVIRONMENT: &[&str] = &[
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_COMMON_DIR",
    "GIT_INDEX_FILE",
    "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_NAMESPACE",
    "GIT_CONFIG_COUNT",
    "GIT_CONFIG_PARAMETERS",
    "GIT_SHALLOW_FILE",
];

struct Work {
    directory: PathBuf,
    started: Instant,
    sequence: usize,
}

struct Output {
    code: u32,
    stdout: Vec<u8>,
    stderr: Vec<u8>,
}

impl Output {
    fn success(self, program: &str) -> Result<Vec<u8>> {
        if self.code == 0 {
            return Ok(self.stdout);
        }
        let mut detail = self.stdout;
        detail.extend(self.stderr);
        let excerpt = String::from_utf8_lossy(&detail[detail.len().saturating_sub(8192)..]);
        Err(format!(
            "Push blocked: {program} exited {}.\n{excerpt}",
            self.code
        ))
    }
}

impl Work {
    fn shell(
        &mut self,
        root: &Path,
        arguments: &[OsString],
        environment: BTreeMap<String, String>,
    ) -> Result<Output> {
        // Packaged PowerShell cannot be created suspended directly. Start
        // its ordinary launcher inside the already-owned, non-breakaway Job.
        let mut words =
            vec![std::env::var_os("PRISM_PUSH_PWSH_PATH").unwrap_or_else(|| "pwsh.exe".into())];
        words.extend_from_slice(arguments);
        let words = words
            .into_iter()
            .map(|word| {
                let word = word
                    .to_str()
                    .ok_or("PowerShell startup paths must be UTF-8.")?;
                if word.contains(['\"', '\r', '\n', '\0']) {
                    return Err("Unsupported PowerShell startup argument.".into());
                }
                Ok(format!("\"{}\"", word.replace('%', "%%")))
            })
            .collect::<Result<Vec<_>>>()?;
        let batch = self.directory.join(format!("shell-{}.cmd", self.sequence));
        fs::write(&batch, format!(
            "@echo off\r\nchcp 65001 >nul\r\nset \"ERRORLEVEL=\"\r\n{}\r\nexit /b %errorlevel%\r\n",
            words.join(" ")
        )).map_err(|error| error.to_string())?;
        self.command(
            root,
            "cmd.exe",
            &["/d".into(), "/c".into(), batch.into_os_string()],
            environment,
        )
    }

    fn command(
        &mut self,
        root: &Path,
        program: &str,
        arguments: &[OsString],
        mut environment: BTreeMap<String, String>,
    ) -> Result<Output> {
        if self.started.elapsed() >= WORK_LIMIT {
            return Err("Push sanity aggregate allowance exhausted; push blocked.".into());
        }
        if program == "pwsh.exe" {
            return self.shell(root, arguments, environment);
        }
        if program == "git.exe" {
            // Repository-local replacement refs must never substitute repaired
            // bytes for the immutable outgoing object actually supplied by Git.
            environment.insert("GIT_NO_REPLACE_OBJECTS".into(), "1".into());
        }
        let executable = std::env::split_paths(&std::env::var_os("PATH").unwrap_or_default())
            .map(|directory| directory.join(program))
            .find(|path| path.is_absolute() && path.is_file())
            .ok_or_else(|| format!("Push sanity requires {program} on PATH."))?;
        let name = self.sequence;
        self.sequence += 1;
        let stdout = self.directory.join(format!("{name}.stdout"));
        let stderr = self.directory.join(format!("{name}.stderr"));
        let removed = REMOVED_GIT_ENVIRONMENT
            .iter()
            .filter(|name| !environment.contains_key(**name))
            .map(|name| (*name).to_owned())
            .collect::<Vec<_>>();
        let child = OwnedProcess::start_tool_with_removed(
            &ToolProcessSpec {
                executable,
                arguments: arguments.into(),
                working_directory: root.into(),
                stdout: stdout.clone(),
                stderr: stderr.clone(),
                environment,
            },
            &removed,
        )
        .map_err(|error| format!("Cannot start {program}: {error}"))?;
        let exceeded_output = || {
            [&stdout, &stderr]
                .into_iter()
                .any(|path| fs::metadata(path).is_ok_and(|metadata| metadata.len() > OUTPUT_LIMIT))
        };
        let code = loop {
            if self.started.elapsed() >= WORK_LIMIT || exceeded_output() {
                child
                    .stop(Duration::from_secs(5))
                    .map_err(|error| error.to_string())?;
                return Err(
                    "Push sanity exceeded its shared allowance or output bound; push blocked."
                        .into(),
                );
            }
            if let Some(code) = child.try_wait().map_err(|error| error.to_string())? {
                break code;
            }
            std::thread::sleep(Duration::from_millis(20));
        };
        child
            .stop(Duration::from_secs(5))
            .map_err(|error| error.to_string())?;
        drop(child);
        if exceeded_output() {
            return Err("Push command output exceeded its final byte bound; push blocked.".into());
        }
        Ok(Output {
            code,
            stdout: fs::read(stdout).map_err(|error| error.to_string())?,
            stderr: fs::read(stderr).map_err(|error| error.to_string())?,
        })
    }

    fn git(&mut self, root: &Path, arguments: &[&str]) -> Result<Vec<u8>> {
        self.command(
            root,
            "git.exe",
            &arguments.iter().map(OsString::from).collect::<Vec<_>>(),
            BTreeMap::new(),
        )?
        .success("git")
    }

    fn commit(&mut self, root: &Path, oid: &str, optional: bool) -> Result<Option<String>> {
        let output = self.command(
            root,
            "git.exe",
            &[
                "rev-parse".into(),
                "--verify".into(),
                "--quiet".into(),
                format!("{oid}^{{commit}}").into(),
            ],
            BTreeMap::new(),
        )?;
        if optional && output.code == 1 {
            return Ok(None);
        }
        let raw = output.success("git rev-parse")?;
        let commit = std::str::from_utf8(&raw)
            .map_err(|error| error.to_string())?
            .trim();
        if !super::oid(commit) {
            return Err("Git returned an invalid commit identity.".into());
        }
        Ok(Some(commit.to_owned()))
    }

    fn commits(
        &mut self,
        root: &Path,
        updates: &[Update],
    ) -> Result<BTreeMap<String, BTreeSet<String>>> {
        let mut commits = BTreeMap::<String, BTreeSet<String>>::new();
        for update in updates {
            let commit = self
                .commit(root, &update.tip, false)?
                .ok_or("Missing outgoing commit.")?;
            let base = if update.base.bytes().all(|byte| byte == b'0') {
                None
            } else {
                self.commit(root, &update.base, true)?
            };
            let changed = if let Some(base) = base {
                self.git(
                    root,
                    &[
                        "diff",
                        "--no-ext-diff",
                        "--no-renames",
                        "--name-only",
                        "-z",
                        &base,
                        &commit,
                        "--",
                    ],
                )?
            } else {
                self.git(root, &["ls-tree", "-r", "--name-only", "-z", &commit])?
            };
            commits.entry(commit).or_default().extend(paths(&changed)?);
        }
        Ok(commits)
    }

    fn snapshot(&mut self, root: &Path, commit: &str, number: usize) -> Result<PathBuf> {
        let source = self.directory.join(format!("source-{number}"));
        fs::create_dir(&source).map_err(|error| error.to_string())?;
        let environment = BTreeMap::from([(
            "GIT_INDEX_FILE".into(),
            self.directory
                .join(format!("index-{number}"))
                .to_string_lossy()
                .into_owned(),
        )]);
        self.command(
            root,
            "git.exe",
            &["read-tree".into(), commit.into()],
            environment.clone(),
        )?
        .success("git read-tree")?;
        self.command(
            root,
            "git.exe",
            &[
                "checkout-index".into(),
                "--all".into(),
                format!("--prefix={}/", source.display()).into(),
            ],
            environment,
        )?
        .success("git checkout-index")?;
        // actionlint discovers local composite-action inputs only in a repository.
        // No live Git metadata is copied into this immutable source snapshot.
        fs::create_dir(source.join(".git")).map_err(|error| error.to_string())?;
        Ok(source)
    }

    fn checks(&mut self, source: &Path, policy: &Path, paths: &[String]) -> Result<()> {
        // Check formatting before calling any interpreter. The candidate contributes
        // source and configuration, while checking scripts come from trusted HEAD.
        if super::rust_selected(paths) {
            let pin = super::toolchain(
                &fs::read_to_string(source.join("rust-toolchain.toml"))
                    .map_err(|error| error.to_string())?,
            )?;
            let environment = BTreeMap::from([
                ("CARGO_NET_OFFLINE".into(), "true".into()),
                ("RUSTUP_AUTO_INSTALL".into(), "0".into()),
            ]);
            let image = self
                .command(
                    source,
                    "rustup.exe",
                    &[
                        "which".into(),
                        "--toolchain".into(),
                        pin.clone().into(),
                        "rustfmt".into(),
                    ],
                    environment.clone(),
                )?
                .success("rustup which rustfmt")?;
            let image = std::str::from_utf8(&image)
                .map_err(|error| error.to_string())?
                .trim();
            if !Path::new(image).is_absolute() || !Path::new(image).is_file() {
                return Err("Pinned rustfmt image is unavailable; push blocked.".into());
            }
            let mut environment = environment;
            environment.insert("RUSTFMT".into(), image.into());
            self.command(
                source,
                "cargo.exe",
                &[
                    format!("+{pin}").into(),
                    "fmt".into(),
                    "--all".into(),
                    "--manifest-path".into(),
                    source.join("Cargo.toml").into_os_string(),
                    "--".into(),
                    "--check".into(),
                ],
                environment,
            )?
            .success("cargo fmt --all --check")?;
        }
        let paths_file = self.directory.join("changed-paths.json");
        fs::write(
            &paths_file,
            serde_json::to_vec(paths).map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
        self.command(
            source,
            "pwsh.exe",
            &[
                "-NoLogo".into(),
                "-NoProfile".into(),
                "-NonInteractive".into(),
                "-File".into(),
                policy
                    .join(".github/ci/Check-SourceSnapshot.ps1")
                    .into_os_string(),
                "-Root".into(),
                source.as_os_str().into(),
                "-PathsFile".into(),
                paths_file.into_os_string(),
            ],
            BTreeMap::new(),
        )?
        .success("trusted Check-SourceSnapshot.ps1")?;
        Ok(())
    }
}

fn paths(raw: &[u8]) -> Result<Vec<String>> {
    raw.split(|byte| *byte == 0)
        .filter(|path| !path.is_empty())
        .map(|path| {
            std::str::from_utf8(path)
                .map(str::to_owned)
                .map_err(|error| error.to_string())
        })
        .collect()
}

#[expect(
    clippy::redundant_pub_crate,
    reason = "The binary implementation exposes only to its parent module."
)]
pub(super) fn check(
    root: &Path,
    policy: &Path,
    updates: &[Update],
    bootstrap: u64,
) -> Result<String> {
    if !cfg!(target_arch = "aarch64") {
        return Err("Push sanity requires native Windows ARM64.".into());
    }
    if Duration::from_millis(bootstrap) >= WORK_LIMIT {
        return Err("Push sanity bootstrap exhausted the shared allowance; push blocked.".into());
    }
    let started = Instant::now()
        .checked_sub(Duration::from_millis(bootstrap))
        .ok_or("Push bootstrap start predates the monotonic clock.")?;
    let temporary = super::canonical_path(std::env::temp_dir())?;
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| error.to_string())?
        .as_nanos();
    let directory = temporary.join(format!("prism-push-sanity-{}-{nonce}", std::process::id()));
    fs::create_dir(&directory).map_err(|error| error.to_string())?;
    let before = bonsai_windows_process::power_status();
    let mut work = Work {
        directory,
        started,
        sequence: 0,
    };
    let result = (|| {
        let commits = work.commits(root, updates)?;
        for (number, (commit, paths)) in commits.iter().enumerate() {
            let source = work.snapshot(root, commit, number)?;
            work.checks(&source, policy, &paths.iter().cloned().collect::<Vec<_>>())?;
        }
        Ok(format!(
            "Push sanity passed for {} immutable outgoing source tip(s); hosted compilation, analyzers and tests remain separate.",
            commits.len()
        ))
    })();
    let cleanup = if work.directory.parent() == Some(temporary.as_path()) {
        fs::remove_dir_all(&work.directory).map_err(|error| error.to_string())
    } else {
        Err("Scratch ownership changed.".into())
    };
    let after = bonsai_windows_process::power_status();
    let _ = writeln!(
        std::io::stderr().lock(),
        "Push sanity native power before={before:?}; after-cleanup={after:?}; elapsed={:?}.",
        work.started.elapsed()
    );
    cleanup.map_err(|error| {
        format!("Push sanity cleanup failed: {error}; original outcome={result:?}")
    })?;
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn with_work(operation: impl FnOnce(&mut Work) -> Result<()>) -> Result<()> {
        let temporary = crate::canonical_path(std::env::temp_dir())?;
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|error| error.to_string())?
            .as_nanos();
        let directory = temporary.join(format!("prism-sanity-fixture-{nonce}"));
        fs::create_dir(&directory).map_err(|error| error.to_string())?;
        let mut work = Work {
            directory: directory.clone(),
            started: Instant::now(),
            sequence: 0,
        };
        let result = operation(&mut work);
        if directory.parent() != Some(temporary.as_path()) {
            return Err("Fixture cleanup escaped temporary storage.".into());
        }
        fs::remove_dir_all(directory).map_err(|error| error.to_string())?;
        result
    }

    fn commit(work: &mut Work, root: &Path, message: &str) -> Result<String> {
        work.git(root, &["add", "--all"])?;
        work.git(
            root,
            &[
                "-c",
                "user.name=Fixture",
                "-c",
                "user.email=fixture@example.invalid",
                "-c",
                "core.hooksPath=",
                "-c",
                "commit.gpgsign=false",
                "commit",
                "--quiet",
                "-m",
                message,
            ],
        )?;
        work.commit(root, "HEAD", false)?
            .ok_or_else(|| "Missing fixture commit.".into())
    }

    #[test]
    fn trusted_adapter_regressions_are_owned_by_foundation() -> Result<()> {
        with_work(|work| {
            let root = crate::canonical_path(Path::new(env!("CARGO_MANIFEST_DIR")).join("../.."))?;
            // Receipt admission is a native ARM64 operation. Other Windows
            // Foundation lanes still exercise parse-only policy isolation.
            let case = if cfg!(target_arch = "aarch64") {
                "all"
            } else {
                "syntax"
            };
            work.command(
                &root,
                "pwsh.exe",
                &[
                    "-NoLogo".into(),
                    "-NoProfile".into(),
                    "-NonInteractive".into(),
                    "-File".into(),
                    root.join(".github/ci/Test-SourceSanity.ps1")
                        .into_os_string(),
                    "-Case".into(),
                    case.into(),
                ],
                BTreeMap::new(),
            )?
            .success("source-sanity adapter regressions")?;
            Ok(())
        })
    }

    struct Fixture {
        root: PathBuf,
        broken: String,
        repaired: String,
        tag: String,
        index: Vec<u8>,
    }

    fn repository(work: &mut Work) -> Result<Fixture> {
        let root = work.directory.join("repository");
        fs::create_dir(&root).map_err(|error| error.to_string())?;
        work.git(&root, &["init", "--quiet"])?;
        fs::write(root.join("file with spaces.ps1"), "function broken {")
            .map_err(|error| error.to_string())?;
        fs::write(root.join("untouched.md"), "baseline").map_err(|error| error.to_string())?;
        let broken = commit(work, &root, "broken")?;
        work.git(
            &root,
            &[
                "-c",
                "user.name=Fixture",
                "-c",
                "user.email=fixture@example.invalid",
                "-c",
                "tag.gpgsign=false",
                "tag",
                "-a",
                "v-broken",
                "-m",
                "broken",
                &broken,
            ],
        )?;
        let tag = String::from_utf8(work.git(&root, &["rev-parse", "v-broken"])?)
            .map_err(|error| error.to_string())?
            .trim()
            .to_owned();
        fs::write(root.join("file with spaces.ps1"), "function repaired {}")
            .map_err(|error| error.to_string())?;
        let repaired = commit(work, &root, "repaired")?;
        fs::write(root.join("file with spaces.ps1"), "staged repair")
            .map_err(|error| error.to_string())?;
        work.git(&root, &["add", "--all"])?;
        fs::write(root.join("file with spaces.ps1"), "unstaged repair")
            .map_err(|error| error.to_string())?;
        let index = work.git(&root, &["ls-files", "--stage", "-z"])?;
        Ok(Fixture {
            root,
            broken,
            repaired,
            tag,
            index,
        })
    }

    #[test]
    fn immutable_tips_peel_tags_union_bases_and_preserve_owner_state() -> Result<()> {
        with_work(|work| {
            let Fixture {
                root,
                broken,
                repaired,
                tag,
                index,
            } = repository(work)?;
            work.git(&root, &["replace", &broken, &repaired])?;
            let commits = work.commits(
                &root,
                &[
                    Update {
                        tip: tag,
                        base: repaired.clone(),
                    },
                    Update {
                        tip: broken.clone(),
                        base: "f".repeat(40),
                    },
                    Update {
                        tip: repaired.clone(),
                        base: broken.clone(),
                    },
                ],
            )?;
            assert_eq!(commits.len(), 2);
            assert!(commits[&broken].contains("untouched.md"));
            assert!(!commits[&repaired].contains("untouched.md"));
            let snapshot = work.snapshot(&root, &broken, 0)?;
            assert_eq!(
                fs::read_to_string(snapshot.join("file with spaces.ps1"))
                    .map_err(|error| error.to_string())?,
                "function broken {"
            );
            assert!(snapshot.join(".git").is_dir());
            assert_eq!(
                fs::read_to_string(root.join("file with spaces.ps1"))
                    .map_err(|error| error.to_string())?,
                "unstaged repair"
            );
            assert_eq!(work.git(&root, &["ls-files", "--stage", "-z"])?, index);
            Ok(())
        })
    }

    #[test]
    fn rejects_non_utf8_git_paths() {
        assert!(paths(&[b'a', 0, 0xff]).is_err());
    }
}
