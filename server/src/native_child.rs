//! Internal native Swift backend entry point for the existing Windows Job owner.

use std::{ffi::OsString, io, path::PathBuf};

use bonsai_prism_runtime::{
    kernels::{MAX_ROW_WORKERS, MatvecBackend, RowParallelism},
    model::swift::Representation,
    swift_session::SwiftSession,
};
use bonsai_windows_credentials::validate_read_path;

use crate::service::bound_loopback_socket;

struct Config {
    model: PathBuf,
    key: PathBuf,
    port: u16,
    seed: u32,
    row_workers: usize,
    representation: Representation,
    adapter: Option<PathBuf>,
}

fn invalid() -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidInput,
        "invalid native child configuration",
    )
}

fn parse(arguments: impl IntoIterator<Item = OsString>) -> io::Result<Config> {
    let mut arguments = arguments.into_iter();
    let mut child = false;
    let (mut model, mut key, mut port, mut seed) = (None, None, None, None);
    let mut row_workers = None;
    let mut repacked_ptq1 = false;
    let mut adapter = None;
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
    };
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
    let config = parse(arguments)?;
    crate::configuration::validate_environment(std::env::vars_os().map(|(name, _)| name))
        .map_err(|_| invalid())?;
    if bonsai_windows_process::available_memory_bytes()?
        < crate::configuration::REQUIRED_AVAILABLE_BYTES
    {
        return Err(io::Error::other(
            "native Swift startup memory admission rejected",
        ));
    }
    let (_key_guard, key) = crate::configuration::read_key(&config.key).map_err(|_| invalid())?;
    let parallelism = RowParallelism::new(config.row_workers).map_err(|_| invalid())?;
    let session = SwiftSession::load_profile_with_row_parallelism(
        &config.model,
        config.representation,
        config.adapter.as_deref(),
        MatvecBackend::Auto,
        config.seed,
        parallelism,
    )
    .map_err(|_| io::Error::other("verified native Swift loading failed"))?;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    runtime.block_on(async {
        let listener = bound_loopback_socket(config.port)?.listen(1)?;
        crate::native_backend::serve(listener, session, key)
            .await
            .map_err(|_| io::Error::other("native Swift backend admission closed"))
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
        for value in ["0", "17", "invalid"] {
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
}
