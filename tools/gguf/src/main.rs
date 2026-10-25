#![forbid(unsafe_code)]
//! Command-line entry point for verified GGUF inspection and PTQ1 export.

use std::collections::BTreeMap;
use std::ffi::OsString;
use std::fs::OpenOptions;
use std::io::{self, Write};
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Duration;

const USAGE: &str = "bonsai-gguf inspect --model PATH --expected-bytes N --expected-sha256 HASH [--dialect Stock|Prism-73bfe6cc485554822f80c473a3c0c4f25e80f41b] [--census-pq2] [--output PATH]\n\
bonsai-gguf export-ptq1 --model PATH --output PATH --expected-bytes N --expected-sha256 HASH [--timeout-seconds 1..3600]\n";

enum Command {
    Inspect,
    ExportPtq1,
}

struct Arguments {
    command: Command,
    model: PathBuf,
    output: Option<PathBuf>,
    expected_bytes: u64,
    expected_sha256: String,
    dialect: String,
    census: bool,
    timeout: Duration,
}

fn required(values: &mut BTreeMap<String, OsString>, flag: &str) -> Result<OsString, String> {
    values
        .remove(flag)
        .ok_or_else(|| format!("Missing required argument {flag}."))
}

fn text(value: OsString, flag: &str) -> Result<String, String> {
    value
        .into_string()
        .map_err(|_| format!("{flag} must contain valid Unicode."))
}

fn parse(mut args: impl Iterator<Item = OsString>) -> Result<Arguments, String> {
    let command = match args.next().and_then(|value| value.into_string().ok()) {
        Some(value) if value == "inspect" => Command::Inspect,
        Some(value) if value == "export-ptq1" => Command::ExportPtq1,
        _ => return Err(format!("Choose inspect or export-ptq1.\n{USAGE}")),
    };
    let mut values = BTreeMap::new();
    let mut census = false;
    while let Some(flag) = args.next() {
        let flag = text(flag, "Option name")?;
        if flag == "--census-pq2" && matches!(command, Command::Inspect) {
            if census {
                return Err(format!("Repeated argument {flag}."));
            }
            census = true;
            continue;
        }
        let valid = matches!(
            flag.as_str(),
            "--model" | "--output" | "--expected-bytes" | "--expected-sha256"
        ) || (flag == "--dialect" && matches!(command, Command::Inspect))
            || (flag == "--timeout-seconds" && matches!(command, Command::ExportPtq1));
        if !valid {
            return Err(format!("Unknown argument {flag}.\n{USAGE}"));
        }
        let value = args
            .next()
            .filter(|value| !value.to_string_lossy().starts_with("--"))
            .ok_or_else(|| format!("{flag} requires a value."))?;
        if values.insert(flag.clone(), value).is_some() {
            return Err(format!("Repeated argument {flag}."));
        }
    }
    let model = PathBuf::from(required(&mut values, "--model")?);
    let output = values.remove("--output").map(PathBuf::from);
    if matches!(command, Command::ExportPtq1) && output.is_none() {
        return Err("Missing required argument --output.".into());
    }
    let expected_bytes = text(
        required(&mut values, "--expected-bytes")?,
        "--expected-bytes",
    )?
    .parse::<u64>()
    .ok()
    .filter(|value| *value > 0)
    .ok_or("--expected-bytes must be a positive integer.")?;
    let expected_sha256 = text(
        required(&mut values, "--expected-sha256")?,
        "--expected-sha256",
    )?;
    if expected_sha256.len() != 64
        || !expected_sha256
            .bytes()
            .all(|value| value.is_ascii_hexdigit())
    {
        return Err("--expected-sha256 must contain exactly 64 hexadecimal digits.".into());
    }
    let dialect = values
        .remove("--dialect")
        .map_or_else(|| Ok("Stock".into()), |value| text(value, "--dialect"))?;
    let timeout = values.remove("--timeout-seconds").map_or_else(
        || Ok(1800),
        |value| {
            text(value, "--timeout-seconds")?
                .parse::<u64>()
                .map_err(|_| "--timeout-seconds must be an integer in 1..3600.".to_owned())
        },
    )?;
    if !(1..=3600).contains(&timeout) {
        return Err("--timeout-seconds must be an integer in 1..3600.".into());
    }
    Ok(Arguments {
        command,
        model,
        output,
        expected_bytes,
        expected_sha256,
        dialect,
        census,
        timeout: Duration::from_secs(timeout),
    })
}

fn run(args: Arguments) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let report = match args.command {
        Command::Inspect => bonsai_gguf::inspect(
            &args.model,
            &args.dialect,
            args.census,
            args.expected_bytes,
            &args.expected_sha256,
        )?,
        Command::ExportPtq1 => bonsai_gguf::export_ptq1(
            &args.model,
            args.output
                .as_deref()
                .ok_or("Missing required argument --output.")?,
            args.expected_bytes,
            &args.expected_sha256,
            args.timeout,
        )?,
    };
    let mut json = serde_json::to_vec_pretty(&report)?;
    json.push(b'\n');
    if matches!(args.command, Command::Inspect)
        && let Some(output) = args.output
    {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(output)?;
        file.write_all(&json)?;
        file.sync_all()?;
    }
    io::stdout().lock().write_all(&json)?;
    Ok(())
}

fn main() -> ExitCode {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.len() == 1 && (args[0] == "--help" || args[0] == "-h") {
        return if io::stdout().lock().write_all(USAGE.as_bytes()).is_ok() {
            ExitCode::SUCCESS
        } else {
            ExitCode::FAILURE
        };
    }
    let result = parse(args.into_iter()).map_err(Into::into).and_then(run);
    if let Err(error) = result {
        let _ = writeln!(io::stderr().lock(), "bonsai-gguf: {error}");
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}

#[cfg(test)]
mod tests {
    use super::{Command, parse};
    use std::ffi::OsString;

    const SHA256: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    fn arguments(command: &str) -> Vec<OsString> {
        [
            command,
            "--model",
            "fixture.gguf",
            "--expected-bytes",
            "4096",
            "--expected-sha256",
            SHA256,
        ]
        .map(OsString::from)
        .to_vec()
    }

    #[test]
    fn rejects_duplicate_and_unknown_flags() {
        let mut args = arguments("inspect");
        args.extend(["--model", "other.gguf"].map(OsString::from));
        assert!(parse(args.into_iter()).is_err());
        let mut args = arguments("inspect");
        args.push(OsString::from("--unknown"));
        assert!(parse(args.into_iter()).is_err());
        let mut args = arguments("inspect");
        args.extend(["--census-pq2", "--census-pq2"].map(OsString::from));
        assert!(parse(args.into_iter()).is_err());
    }

    #[test]
    fn rejects_incomplete_and_invalid_pin() {
        let mut args = arguments("inspect");
        args.truncate(args.len() - 2);
        assert!(parse(args.into_iter()).is_err());
        let mut args = arguments("inspect");
        args[4] = OsString::from("0");
        assert!(parse(args.into_iter()).is_err());
        let mut args = arguments("inspect");
        args[6] = OsString::from("not-a-sha256");
        assert!(parse(args.into_iter()).is_err());
    }

    #[test]
    fn requires_export_output_and_bounds_timeout() {
        assert!(parse(arguments("export-ptq1").into_iter()).is_err());
        for timeout in ["0", "3601", "-1", "invalid"] {
            let mut args = arguments("export-ptq1");
            args.extend(
                ["--output", "export.gguf", "--timeout-seconds", timeout].map(OsString::from),
            );
            assert!(parse(args.into_iter()).is_err());
        }
    }

    #[test]
    fn accepts_timeout_boundaries_and_inspect_defaults() -> Result<(), String> {
        for timeout in ["1", "3600"] {
            let mut args = arguments("export-ptq1");
            args.extend(
                ["--output", "export.gguf", "--timeout-seconds", timeout].map(OsString::from),
            );
            let parsed = parse(args.into_iter())?;
            assert!(matches!(parsed.command, Command::ExportPtq1));
        }
        let parsed = parse(arguments("inspect").into_iter())?;
        assert_eq!(parsed.dialect, "Stock");
        assert!(!parsed.census);
        assert!(parsed.output.is_none());
        Ok(())
    }

    #[test]
    fn delegates_census_dialect_validation_to_library() -> Result<(), String> {
        let mut args = arguments("inspect");
        args.push(OsString::from("--census-pq2"));
        let parsed = parse(args.into_iter())?;
        assert!(parsed.census);
        assert_eq!(parsed.dialect, "Stock");
        Ok(())
    }
}
