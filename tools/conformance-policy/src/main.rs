//! JSON command entry point for Prism conformance policy.
#![forbid(unsafe_code)]

use std::fs::OpenOptions;
use std::io::{Read as _, Write as _};
use std::path::Path;
use std::process::ExitCode;

const MAX_REQUEST_BYTES: u64 = 16 * 1024 * 1024;
const COMMANDS: &[&str] = &["prism-source", "prism-baseline"];

fn read_request(path: &Path, maximum: u64) -> prism_conformance_policy::Result<serde_json::Value> {
    if !path.is_absolute() {
        return Err("Request path must be expanded and absolute.".into());
    }
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt as _;
        options.share_mode(1);
    }
    let file = options.open(path).map_err(|error| error.to_string())?;
    if file.metadata().map_err(|error| error.to_string())?.len() > maximum {
        return Err(format!("Command request exceeds {maximum} bytes."));
    }
    let mut bytes = Vec::new();
    file.take(maximum + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| error.to_string())?;
    if u64::try_from(bytes.len()).map_err(|error| error.to_string())? > maximum {
        return Err(format!("Command request exceeds {maximum} bytes."));
    }
    let request: serde_json::Value =
        serde_json::from_slice(&bytes).map_err(|error| error.to_string())?;
    if !request.is_object() {
        return Err("Command request must be a JSON object.".into());
    }
    Ok(request)
}

fn execute() -> prism_conformance_policy::Result<serde_json::Value> {
    let arguments: Vec<_> = std::env::args_os().skip(1).collect();
    if arguments.len() == 1 && arguments[0] == "--version" {
        return Ok(serde_json::json!({
            "Name": "prism-conformance-policy", "Version": env!("CARGO_PKG_VERSION"),
            "ProtocolVersion": 1, "Commands": COMMANDS
        }));
    }
    if arguments.len() != 3 || arguments[1] != "--request" {
        return Err("Usage: prism-conformance-policy <prism-source|prism-baseline> --request ABSOLUTE_JSON_PATH; or --version".into());
    }
    let request = read_request(Path::new(&arguments[2]), MAX_REQUEST_BYTES)?;
    match arguments[0].to_str() {
        Some("prism-source") => prism_conformance_policy::prism_source::run(request),
        Some("prism-baseline") => prism_conformance_policy::prism_baseline::run(request),
        _ => Err("Unknown conformance policy command.".into()),
    }
}

fn main() -> ExitCode {
    match execute() {
        Ok(value) => {
            let mut output = std::io::stdout().lock();
            if serde_json::to_writer(&mut output, &value).is_err()
                || output.write_all(b"\n").is_err()
            {
                return ExitCode::FAILURE;
            }
            ExitCode::SUCCESS
        }
        Err(error) => {
            let _ = writeln!(std::io::stderr().lock(), "{error}");
            ExitCode::FAILURE
        }
    }
}
