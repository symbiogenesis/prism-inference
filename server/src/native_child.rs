//! Internal native Swift backend entry point for the existing Windows Job owner.

use std::{ffi::OsString, io, path::PathBuf};

use bonsai_prism_runtime::{kernels::MatvecBackend, swift_session::SwiftSession};
use bonsai_windows_credentials::validate_external_path;

use crate::service::bound_loopback_socket;

struct Config {
    model: PathBuf,
    key: PathBuf,
    port: u16,
    seed: u32,
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
    while let Some(argument) = arguments.next() {
        if argument == "--native-engine-child" && !child {
            child = true;
            continue;
        }
        let value = arguments.next().ok_or_else(invalid)?;
        match argument.to_str() {
            Some("--model") if model.is_none() => model = Some(PathBuf::from(value)),
            Some("--key-file") if key.is_none() => key = Some(PathBuf::from(value)),
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
            _ => return Err(invalid()),
        }
    }
    let config = Config {
        model: model.ok_or_else(invalid)?,
        key: key.ok_or_else(invalid)?,
        port: port.filter(|&n| n >= 1024).ok_or_else(invalid)?,
        seed: seed.filter(|&n| n != u32::MAX).ok_or_else(invalid)?,
    };
    if !child {
        return Err(invalid());
    }
    for path in [&config.model, &config.key] {
        validate_external_path(path).map_err(|_| invalid())?;
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
    let session = SwiftSession::load(&config.model, MatvecBackend::Auto, config.seed)
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

    #[test]
    fn native_child_rejects_unknown_duplicate_relative_and_unbounded_options() {
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
        let mut good = good;
        good[2] = std::env::temp_dir()
            .join("native-model.gguf")
            .into_os_string();
        good[6] = std::env::temp_dir()
            .join("native-backend.key")
            .into_os_string();
        assert!(parse(good.clone()).is_ok());
        for extra in [
            vec!["--seed", "1"],
            vec!["--profile", "swift-pq2-assessment"],
            vec!["--native-engine-child"],
        ] {
            let mut args = good.clone().to_vec();
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
    }
}
