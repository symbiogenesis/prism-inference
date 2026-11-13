//! Internal native Swift backend entry point for the existing Windows Job owner.

use std::{ffi::OsString, io, path::PathBuf};

use bonsai_prism_runtime::{
    graph::KvCachePolicy,
    kernels::{MAX_ROW_WORKERS, MatvecBackend, RowParallelism},
    model::swift::Representation,
    runtime::PromptReuse,
    swift_session::{
        CpuPrefill, CpuSpeculation, SwiftSession, validate_context_capacity,
        validate_execution_capacity,
    },
};
use bonsai_windows_credentials::validate_read_path;

use crate::{
    service::bound_loopback_socket,
    startup::{self, Stage},
};

struct Config {
    model: PathBuf,
    key: PathBuf,
    port: u16,
    seed: u32,
    row_workers: usize,
    representation: Representation,
    adapter: Option<PathBuf>,
    prompt_reuse: PromptReuse,
    kv_cache: KvCachePolicy,
    cpu_prefill: Option<CpuPrefill>,
    cpu_speculation: Option<CpuSpeculation>,
    capacity: Option<(usize, u64)>,
}

fn invalid() -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidInput,
        "invalid native child configuration",
    )
}

#[expect(
    clippy::too_many_lines,
    reason = "Keep trusted profile parsing and pre-I/O memory admission visibly ordered."
)]
fn parse(arguments: impl IntoIterator<Item = OsString>) -> io::Result<Config> {
    let mut arguments = arguments.into_iter();
    let mut child = false;
    let (mut model, mut key, mut port, mut seed) = (None, None, None, None);
    let mut row_workers = None;
    let mut repacked_ptq1 = false;
    let mut adapter = None;
    let mut prompt_reuse = None;
    let mut kv_cache = None;
    let (mut prefill_tokens, mut prefill_scratch) = (None, None);
    let (mut speculative_tokens, mut verification_scratch, mut speculative_checkpoint) =
        (None, None, None);
    let (mut context_tokens, mut memory_limit_bytes) = (None, None);
    while let Some(argument) = arguments.next() {
        if argument == "--native-engine-child" && !child {
            child = true;
            continue;
        }
        if argument == "--use-repacked-ptq1" && !repacked_ptq1 {
            repacked_ptq1 = true;
            continue;
        }
        let value = arguments.next().ok_or_else(invalid)?;
        match argument.to_str() {
            Some("--model") if model.is_none() => model = Some(PathBuf::from(value)),
            Some("--key-file") if key.is_none() => key = Some(PathBuf::from(value)),
            Some("--abliterate-lora") if adapter.is_none() => adapter = Some(PathBuf::from(value)),
            Some("--context-tokens") if context_tokens.is_none() => context_tokens = Some(value),
            Some("--memory-limit-bytes") if memory_limit_bytes.is_none() => {
                memory_limit_bytes = Some(value)
            }
            Some("--cpu-prefill-tokens") if prefill_tokens.is_none() => {
                prefill_tokens = Some(value)
            }
            Some("--cpu-prefill-scratch-bytes") if prefill_scratch.is_none() => {
                prefill_scratch = Some(value)
            }
            Some("--cpu-speculative-tokens") if speculative_tokens.is_none() => {
                speculative_tokens = Some(value)
            }
            Some("--cpu-verification-scratch-bytes") if verification_scratch.is_none() => {
                verification_scratch = Some(value)
            }
            Some("--cpu-speculative-checkpoint-bytes") if speculative_checkpoint.is_none() => {
                speculative_checkpoint = Some(value)
            }
            Some("--port") if port.is_none() => {
                port = Some(
                    value
                        .to_str()
                        .ok_or_else(invalid)?
                        .parse::<u16>()
                        .map_err(|_| invalid())?,
                );
            }
            Some("--seed") if seed.is_none() => {
                seed = Some(
                    value
                        .to_str()
                        .ok_or_else(invalid)?
                        .parse::<u32>()
                        .map_err(|_| invalid())?,
                );
            }
            Some("--row-workers") if row_workers.is_none() => {
                row_workers = Some(
                    value
                        .to_str()
                        .ok_or_else(invalid)?
                        .parse::<usize>()
                        .map_err(|_| invalid())?,
                );
            }
            Some("--prompt-reuse") if prompt_reuse.is_none() => {
                prompt_reuse = Some(match value.to_str() {
                    Some("disabled") => PromptReuse::Disabled,
                    Some("latest") => PromptReuse::Latest,
                    _ => return Err(invalid()),
                });
            }
            Some("--kv-cache") if kv_cache.is_none() => {
                kv_cache = Some(match value.to_str() {
                    Some("f16") => KvCachePolicy::F16,
                    Some("rsq4") => KvCachePolicy::Rotated4,
                    Some("rsq4k3v") => KvCachePolicy::Rotated4Keys3Values,
                    _ => return Err(invalid()),
                });
            }
            _ => return Err(invalid()),
        }
    }
    let config = Config {
        model: model.ok_or_else(invalid)?,
        key: key.ok_or_else(invalid)?,
        port: port.filter(|&n| n >= 1024).ok_or_else(invalid)?,
        seed: seed.filter(|&n| n != u32::MAX).ok_or_else(invalid)?,
        row_workers: row_workers.unwrap_or(1),
        representation: if repacked_ptq1 {
            Representation::CurrentRepackedPtq1
        } else {
            Representation::Pq2
        },
        adapter,
        prompt_reuse: prompt_reuse.unwrap_or_default(),
        kv_cache: kv_cache.unwrap_or(bonsai_prism_runtime::swift_session::DEFAULT_KV_CACHE),
        cpu_prefill: parse_prefill(prefill_tokens, prefill_scratch)?,
        cpu_speculation: parse_speculation(
            speculative_tokens,
            verification_scratch,
            speculative_checkpoint,
        )?,
        capacity: parse_capacity(context_tokens, memory_limit_bytes)?,
    };
    if config.cpu_prefill.is_some() || config.cpu_speculation.is_some() {
        let (context, memory) = config.capacity.ok_or_else(invalid)?;
        if kv_cache.is_none() {
            return Err(invalid());
        }
        validate_execution_capacity(context, memory, config.cpu_prefill, config.cpu_speculation)
            .map_err(|_| invalid())?;
    }
    if !(1..=MAX_ROW_WORKERS).contains(&config.row_workers)
        || (config.row_workers > 1 && !cfg!(feature = "parallel-rows"))
    {
        return Err(invalid());
    }
    if !child {
        return Err(invalid());
    }
    for path in [&config.model, &config.key]
        .into_iter()
        .chain(config.adapter.as_ref())
    {
        validate_read_path(path)?;
    }
    Ok(config)
}

fn parse_prefill(
    tokens: Option<OsString>,
    scratch: Option<OsString>,
) -> io::Result<Option<CpuPrefill>> {
    match (tokens, scratch) {
        (None, None) => Ok(None),
        (Some(tokens), Some(scratch)) => {
            let tokens = tokens
                .to_str()
                .ok_or_else(invalid)?
                .parse::<usize>()
                .map_err(|_| invalid())?;
            let scratch = scratch
                .to_str()
                .ok_or_else(invalid)?
                .parse::<usize>()
                .map_err(|_| invalid())?;
            CpuPrefill::new(tokens, scratch)
                .map(Some)
                .map_err(|_| invalid())
        }
        _ => Err(invalid()),
    }
}

fn parse_speculation(
    tokens: Option<OsString>,
    scratch: Option<OsString>,
    checkpoint: Option<OsString>,
) -> io::Result<Option<CpuSpeculation>> {
    match (tokens, scratch, checkpoint) {
        (None, None, None) => Ok(None),
        (Some(tokens), Some(scratch), Some(checkpoint)) => {
            let parse_number = |value: OsString| -> io::Result<usize> {
                value
                    .to_str()
                    .ok_or_else(invalid)?
                    .parse()
                    .map_err(|_| invalid())
            };
            CpuSpeculation::new(
                parse_number(tokens)?,
                parse_number(scratch)?,
                parse_number(checkpoint)?,
            )
            .map(Some)
            .map_err(|_| invalid())
        }
        _ => Err(invalid()),
    }
}

fn parse_capacity(
    context_tokens: Option<OsString>,
    memory_limit_bytes: Option<OsString>,
) -> io::Result<Option<(usize, u64)>> {
    match (context_tokens, memory_limit_bytes) {
        (None, None) => Ok(None),
        (Some(context), Some(memory)) => {
            let context = context
                .to_str()
                .ok_or_else(invalid)?
                .parse::<usize>()
                .map_err(|_| invalid())?;
            let memory = memory
                .to_str()
                .ok_or_else(invalid)?
                .parse::<u64>()
                .map_err(|_| invalid())?;
            if context < 4_352
                || !context.is_multiple_of(256)
                || memory.checked_add(4 * 1024 * 1024 * 1024).is_none()
            {
                return Err(invalid());
            }
            validate_context_capacity(context, memory).map_err(|_| invalid())?;
            Ok(Some((context, memory)))
        }
        _ => Err(invalid()),
    }
}

/// Start only the private backend selected by `--native-engine-child`.
///
/// The Windows development launcher supplies its protected key, external model,
/// inherited environment and Job ownership. Loading precedes listener creation;
/// the parent retains artifact/executable guards and monitors memory and PID.
///
/// # Errors
/// Rejects arguments, environment overrides, low startup memory, key access,
/// verified model loading, listener setup and backend lifecycle failures.
pub fn run(arguments: impl IntoIterator<Item = OsString>) -> io::Result<()> {
    let config = parse(arguments).map_err(|error| startup::at(Stage::Arguments, &error))?;
    crate::native_backend::register_hexagon_model_path(&config.model);
    crate::configuration::validate_environment(std::env::vars_os().map(|(name, _)| name))
        .map_err(|error| startup::at(Stage::Environment, &error))?;
    let required_available_bytes = config.capacity.map_or(
        crate::configuration::REQUIRED_AVAILABLE_BYTES,
        |(_, memory)| memory + 4 * 1024 * 1024 * 1024,
    );
    if bonsai_windows_process::available_memory_bytes()
        .map_err(|error| startup::at(Stage::Memory, &error))?
        < required_available_bytes
    {
        return Err(startup::at(
            Stage::Memory,
            &io::Error::other("native Swift startup memory admission rejected"),
        ));
    }
    let (_key_guard, key) = crate::configuration::read_key(&config.key)
        .map_err(|error| startup::at(Stage::Key, &error))?;
    let parallelism = RowParallelism::new(config.row_workers)
        .map_err(|error| startup::at(Stage::Arguments, error.as_ref()))?;
    let mut session = SwiftSession::load_profile_with_execution(
        &config.model,
        config.representation,
        config.adapter.as_deref(),
        MatvecBackend::Auto,
        config.seed,
        parallelism,
        config.kv_cache,
        config.capacity.map_or(
            bonsai_prism_runtime::model::swift::CONTEXT_TOKENS,
            |(tokens, _)| tokens,
        ),
        config.cpu_prefill,
        config.cpu_speculation,
    )
    .map_err(|error| startup::at(Stage::Model, error.as_ref()))?;
    session
        .set_prompt_reuse(config.prompt_reuse)
        .map_err(|error| startup::at(Stage::Model, error.as_ref()))?;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|error| startup::at(Stage::Runtime, &error))?;
    runtime.block_on(async {
        let listener = bound_loopback_socket(config.port)
            .and_then(|socket| socket.listen(1))
            .map_err(|error| startup::at(Stage::Listener, &error))?;
        crate::native_backend::serve(listener, session, key)
            .await
            .map_err(|_| {
                startup::at(
                    Stage::Backend,
                    &io::Error::other("native Swift backend admission closed"),
                )
            })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn files() -> io::Result<(bonsai_windows_credentials::PrivateDirectory, [OsString; 3])> {
        let directory =
            bonsai_windows_credentials::PrivateDirectory::create(&std::env::temp_dir())?;
        let paths =
            ["model.gguf", "backend.key", "adapter.gguf"].map(|name| directory.path().join(name));
        for path in &paths {
            std::fs::write(path, [])?;
        }
        Ok((directory, paths.map(PathBuf::into_os_string)))
    }

    #[test]
    fn native_child_context_capacity_is_paired_and_checked() -> io::Result<()> {
        use bonsai_prism_runtime::swift_session::{
            CONTEXT_TOKEN_MEMORY_BYTES, MAX_CONTEXT_TOKENS, NON_CONTEXT_MEMORY_BYTES,
        };
        assert_eq!(parse_capacity(None, None)?, None);
        for tokens in [4_352, 8_192, 32_768, MAX_CONTEXT_TOKENS] {
            let memory = NON_CONTEXT_MEMORY_BYTES
                + u64::try_from(tokens).map_err(io::Error::other)? * CONTEXT_TOKEN_MEMORY_BYTES;
            assert_eq!(
                parse_capacity(
                    Some(tokens.to_string().into()),
                    Some(memory.to_string().into())
                )?,
                Some((tokens, memory)),
            );
            assert!(
                parse_capacity(
                    Some(tokens.to_string().into()),
                    Some((memory - 1).to_string().into())
                )
                .is_err()
            );
        }
        for (tokens, memory) in [
            (Some("8192"), None),
            (None, Some("20000000000")),
            (Some("0"), Some("20000000000")),
            (Some("4351"), Some("20000000000")),
            (Some("8193"), Some("20000000000")),
            (Some("63232"), Some("20000000000")),
            (Some("8192"), Some("18446744073709551615")),
        ] {
            assert!(
                parse_capacity(tokens.map(OsString::from), memory.map(OsString::from)).is_err()
            );
        }
        Ok(())
    }

    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "One admission control retains valid tile/reuse profiles beside all paired-option rejections."
    )]
    fn native_child_cpu_prefill_is_paired_and_memory_admitted() -> io::Result<()> {
        use bonsai_prism_runtime::swift_session::{
            CONTEXT_TOKEN_MEMORY_BYTES, NON_CONTEXT_MEMORY_BYTES,
        };
        let (_directory, [model, key, _adapter]) = files()?;
        let base = vec![
            OsString::from("--native-engine-child"),
            "--model".into(),
            model,
            "--key-file".into(),
            key,
            "--port".into(),
            "8081".into(),
            "--seed".into(),
            "42".into(),
        ];
        assert!(parse(base.clone())?.cpu_prefill.is_none());
        let reserve = 64 * 1024 * 1024;
        let memory = NON_CONTEXT_MEMORY_BYTES + 8_192 * CONTEXT_TOKEN_MEMORY_BYTES + reserve;
        for tokens in ["8", "16", "32"] {
            let mut selected = base.clone();
            selected.extend(
                [
                    "--kv-cache",
                    "f16",
                    "--prompt-reuse",
                    "latest",
                    "--context-tokens",
                    "8192",
                    "--memory-limit-bytes",
                    &memory.to_string(),
                    "--cpu-prefill-tokens",
                    tokens,
                    "--cpu-prefill-scratch-bytes",
                    &reserve.to_string(),
                ]
                .map(OsString::from),
            );
            let config = parse(selected.clone())?;
            assert_eq!(
                config.cpu_prefill.map(CpuPrefill::tokens),
                Some(tokens.parse().map_err(io::Error::other)?)
            );
            assert_eq!(
                config.cpu_prefill.map(CpuPrefill::scratch_bytes),
                Some(usize::try_from(reserve).map_err(io::Error::other)?)
            );
            assert_eq!(config.prompt_reuse, PromptReuse::Latest);
            assert_eq!(config.kv_cache, KvCachePolicy::F16);
            for cache in ["rsq4", "rsq4k3v"] {
                let mut rotated = selected.clone();
                rotated[10] = cache.into();
                assert_eq!(parse(rotated)?.kv_cache.name(), cache);
            }
            let mut insufficient = selected.clone();
            insufficient[16] = OsString::from((memory - 1).to_string());
            assert!(parse(insufficient).is_err());
            selected.extend(["--cpu-prefill-tokens", tokens].map(OsString::from));
            assert!(parse(selected).is_err());
        }
        for options in [
            vec!["--cpu-prefill-tokens", "8"],
            vec!["--cpu-prefill-scratch-bytes", "67108864"],
            vec![
                "--cpu-prefill-tokens",
                "8",
                "--cpu-prefill-scratch-bytes",
                "67108864",
            ],
            vec![
                "--kv-cache",
                "f16",
                "--cpu-prefill-tokens",
                "8",
                "--cpu-prefill-scratch-bytes",
                "67108864",
            ],
            vec![
                "--context-tokens",
                "8192",
                "--memory-limit-bytes",
                "20000000000",
                "--cpu-prefill-tokens",
                "8",
                "--cpu-prefill-scratch-bytes",
                "67108864",
            ],
            vec![
                "--kv-cache",
                "f16",
                "--context-tokens",
                "8192",
                "--memory-limit-bytes",
                "20000000000",
                "--cpu-prefill-tokens",
                "7",
                "--cpu-prefill-scratch-bytes",
                "67108864",
            ],
            vec![
                "--kv-cache",
                "f16",
                "--context-tokens",
                "8192",
                "--memory-limit-bytes",
                "20000000000",
                "--cpu-prefill-tokens",
                "8",
                "--cpu-prefill-scratch-bytes",
                "0",
            ],
        ] {
            let mut arguments = base.clone();
            arguments.extend(options.into_iter().map(OsString::from));
            assert!(parse(arguments).is_err());
        }
        Ok(())
    }

    fn speculation_arguments() -> io::Result<(
        bonsai_windows_credentials::PrivateDirectory,
        Vec<OsString>,
        u64,
    )> {
        use bonsai_prism_runtime::swift_session::{
            CONTEXT_TOKEN_MEMORY_BYTES, NON_CONTEXT_MEMORY_BYTES, STATE_MEMORY_BYTES,
        };
        let (directory, [model, key, _adapter]) = files()?;
        let tile = 64 * 1024 * 1024;
        let logits = 16 * 1024 * 1024;
        let checkpoint = STATE_MEMORY_BYTES;
        let memory = NON_CONTEXT_MEMORY_BYTES
            + 8_192 * CONTEXT_TOKEN_MEMORY_BYTES
            + u64::try_from(tile + logits + checkpoint).map_err(io::Error::other)?;
        let base = vec![
            OsString::from("--native-engine-child"),
            "--model".into(),
            model,
            "--key-file".into(),
            key,
            "--port".into(),
            "8081".into(),
            "--seed".into(),
            "42".into(),
            "--kv-cache".into(),
            "f16".into(),
            "--context-tokens".into(),
            "8192".into(),
            "--memory-limit-bytes".into(),
            memory.to_string().into(),
            "--cpu-prefill-tokens".into(),
            "8".into(),
            "--cpu-prefill-scratch-bytes".into(),
            tile.to_string().into(),
        ];
        Ok((directory, base, memory))
    }

    #[test]
    fn native_child_cpu_speculation_requires_complete_triple_and_both_reserves() -> io::Result<()> {
        let (_directory, base, memory) = speculation_arguments()?;
        let logits = 16 * 1024 * 1024;
        let checkpoint = bonsai_prism_runtime::swift_session::STATE_MEMORY_BYTES;
        assert!(parse(base.clone())?.cpu_speculation.is_none());
        for tokens in [1, 4, 8] {
            let mut selected = base.clone();
            selected.extend([
                "--cpu-speculative-tokens".into(),
                tokens.to_string().into(),
                "--cpu-verification-scratch-bytes".into(),
                logits.to_string().into(),
                "--cpu-speculative-checkpoint-bytes".into(),
                checkpoint.to_string().into(),
            ]);
            for cache in ["f16", "rsq4", "rsq4k3v"] {
                selected[10] = cache.into();
                let config = parse(selected.clone())?;
                let speculation = config.cpu_speculation.ok_or_else(invalid)?;
                assert_eq!(speculation.tokens(), tokens);
                assert_eq!(speculation.verification_scratch_bytes(), logits);
                assert_eq!(speculation.checkpoint_bytes(), checkpoint);
                assert_eq!(config.kv_cache.name(), cache);
            }
            let mut insufficient = selected.clone();
            insufficient[14] = (memory - 1).to_string().into();
            assert!(parse(insufficient).is_err());
            let mut no_tiles = selected.clone();
            no_tiles.drain(15..19);
            assert!(parse(no_tiles).is_err());
            let mut small_checkpoint = selected.clone();
            small_checkpoint[24] = (checkpoint - 1).to_string().into();
            assert!(parse(small_checkpoint).is_err());
            selected.extend(["--cpu-speculative-tokens".into(), "4".into()]);
            assert!(parse(selected).is_err());
        }
        for (tokens, scratch, checkpoint) in [
            (None, Some("1"), Some("1073741824")),
            (Some("4"), None, Some("1073741824")),
            (Some("4"), Some("1"), None),
            (Some("0"), Some("1"), Some("1073741824")),
            (Some("9"), Some("1"), Some("1073741824")),
            (Some("4"), Some("0"), Some("1073741824")),
            (Some("4"), Some("1"), Some("0")),
        ] {
            assert!(
                parse_speculation(
                    tokens.map(OsString::from),
                    scratch.map(OsString::from),
                    checkpoint.map(OsString::from)
                )
                .is_err()
            );
        }
        Ok(())
    }

    #[test]
    fn native_startup_rejects_missing_paths_without_loading_or_leaking_values() -> io::Result<()> {
        let (directory, [_model, key, _adapter]) = files()?;
        let arguments = [
            OsString::from("--native-engine-child"),
            "--model".into(),
            directory
                .path()
                .join("PRIVATE_MISSING_MODEL.gguf")
                .into_os_string(),
            "--key-file".into(),
            key,
            "--port".into(),
            "8081".into(),
            "--seed".into(),
            "42".into(),
        ];
        let error = run(arguments)
            .err()
            .ok_or_else(|| io::Error::other("missing file was accepted"))?;
        let mut output = Vec::new();
        startup::write_failure(&mut output, &error)?;
        let output = String::from_utf8(output).map_err(io::Error::other)?;
        assert!(output.starts_with(
            "private-inference-startup-failed stage=arguments errorKind=NotFound osCode="
        ));
        assert!(!output.contains("PRIVATE_MISSING_MODEL"));
        Ok(())
    }

    #[test]
    fn native_child_rejects_unknown_duplicate_relative_and_unbounded_options() -> io::Result<()> {
        let good = [
            "--native-engine-child",
            "--model",
            "C:\\External\\model.gguf",
            "--port",
            "8081",
            "--key-file",
            "C:\\External\\backend.key",
            "--seed",
            "42",
        ]
        .map(OsString::from);
        let (_directory, [model, key, _adapter]) = files()?;
        let mut good = good;
        good[2] = model;
        good[6] = key;
        assert!(parse(good.clone()).is_ok());
        assert_eq!(parse(good.clone())?.row_workers, 1);
        assert_eq!(parse(good.clone())?.prompt_reuse, PromptReuse::Disabled);
        for (value, expected) in [
            ("disabled", PromptReuse::Disabled),
            ("latest", PromptReuse::Latest),
        ] {
            let mut selected = good.to_vec();
            selected.extend(["--prompt-reuse", value].map(OsString::from));
            assert_eq!(parse(selected.clone())?.prompt_reuse, expected);
            selected.extend(["--prompt-reuse", value].map(OsString::from));
            assert!(parse(selected).is_err());
        }
        for value in ["", "true", "Latest", "snapshot", "approximate"] {
            let mut selected = good.to_vec();
            selected.extend(["--prompt-reuse", value].map(OsString::from));
            assert!(parse(selected).is_err());
        }
        for value in ["0", "19", "invalid"] {
            let mut args = good.to_vec();
            args.extend(["--row-workers", value].map(OsString::from));
            assert!(parse(args).is_err());
        }
        let mut parallel = good.to_vec();
        parallel.extend(["--row-workers", "2"].map(OsString::from));
        assert_eq!(parse(parallel).is_ok(), cfg!(feature = "parallel-rows"));
        for extra in [
            vec!["--seed", "1"],
            vec!["--profile", "swift-pq2-assessment"],
            vec!["--native-engine-child"],
        ] {
            let mut args = good.to_vec();
            args.extend(extra.into_iter().map(OsString::from));
            assert!(parse(args).is_err());
        }
        let mut args = good.clone();
        args[2] = OsString::from("relative.gguf");
        assert!(parse(args).is_err());
        for value in ["0", "1023", "65536"] {
            let mut args = good.clone();
            args[4] = OsString::from(value);
            assert!(parse(args).is_err());
        }
        let mut args = good;
        args[8] = OsString::from(u32::MAX.to_string());
        assert!(parse(args).is_err());
        Ok(())
    }

    #[test]
    fn native_child_selects_ptq1_and_approved_adapter_only_with_explicit_options() -> io::Result<()>
    {
        let (directory, [model, key, adapter]) = files()?;
        let base = vec![
            OsString::from("--native-engine-child"),
            "--model".into(),
            model,
            "--key-file".into(),
            key,
            "--port".into(),
            "8081".into(),
            "--seed".into(),
            "42".into(),
        ];
        let default = parse(base.clone())?;
        assert_eq!(default.representation, Representation::Pq2);
        assert!(default.adapter.is_none());
        let mut missing = base.clone();
        missing[2] = directory.path().join("missing.gguf").into_os_string();
        assert!(parse(missing).is_err());
        let mut nonregular = base.clone();
        nonregular[2] = directory.path().as_os_str().to_owned();
        assert!(parse(nonregular).is_err());
        for repacked in [false, true] {
            let mut selected = base.clone();
            if repacked {
                selected.push("--use-repacked-ptq1".into());
            }
            selected.extend([OsString::from("--abliterate-lora"), adapter.clone()]);
            let config = parse(selected.clone())?;
            assert_eq!(
                config.representation,
                if repacked {
                    Representation::CurrentRepackedPtq1
                } else {
                    Representation::Pq2
                }
            );
            assert_eq!(config.adapter, Some(PathBuf::from(&adapter)));
            selected.extend([OsString::from("--abliterate-lora"), adapter.clone()]);
            assert!(parse(selected).is_err());
        }
        for extra in [
            vec!["--use-repacked-ptq1", "--use-repacked-ptq1"],
            vec!["--abliterate-lora", "relative.gguf"],
            vec!["--abliteration-scale", "1.3"],
        ] {
            let mut invalid = base.clone();
            invalid.extend(extra.into_iter().map(OsString::from));
            assert!(parse(invalid).is_err());
        }
        Ok(())
    }

    #[test]
    fn native_child_kv_cache_defaults_to_k4v4_with_explicit_unique_recovery() -> io::Result<()> {
        let (_directory, [model, key, _adapter]) = files()?;
        let base = vec![
            OsString::from("--native-engine-child"),
            "--model".into(),
            model,
            "--key-file".into(),
            key,
            "--port".into(),
            "8081".into(),
            "--seed".into(),
            "42".into(),
        ];
        assert_eq!(parse(base.clone())?.kv_cache, KvCachePolicy::Rotated4);
        for (name, expected) in [
            ("f16", KvCachePolicy::F16),
            ("rsq4", KvCachePolicy::Rotated4),
            ("rsq4k3v", KvCachePolicy::Rotated4Keys3Values),
        ] {
            let mut selected = base.clone();
            selected.extend(["--kv-cache", name].map(OsString::from));
            assert_eq!(parse(selected.clone())?.kv_cache, expected);
            selected.extend(["--kv-cache", name].map(OsString::from));
            assert!(parse(selected).is_err());
        }
        for name in ["", "F16", "q4_0", "turboquant", "rsq3", "rsq4 "] {
            let mut selected = base.clone();
            selected.extend(["--kv-cache", name].map(OsString::from));
            assert!(parse(selected).is_err());
        }
        Ok(())
    }
}
