//! Synthetic native consumer evidence for the exact released Prism codec ABI.
#![deny(unsafe_code)]

use std::io::Write as _;
use std::path::Path;
use std::process::ExitCode;

#[cfg(all(windows, target_arch = "aarch64"))]
#[allow(
    unsafe_code,
    reason = "reviewed dynamic-library boundary is confined to this module"
)]
mod native;
#[cfg(all(windows, target_arch = "aarch64"))]
mod vectors;

#[cfg(all(windows, target_arch = "aarch64"))]
mod supervise;

const RUNTIME_COMMIT: &str = "73bfe6cc485554822f80c473a3c0c4f25e80f41b";

fn run_native(runtime: &Path) -> Result<usize, String> {
    #[cfg(all(windows, target_arch = "aarch64"))]
    {
        native::run(runtime)
    }
    #[cfg(not(all(windows, target_arch = "aarch64")))]
    {
        let _ = runtime;
        Err("The pinned native codec probe requires Windows ARM64.".into())
    }
}

fn execute() -> Result<serde_json::Value, String> {
    let mut arguments: Vec<_> = std::env::args_os().skip(1).collect();
    let native_child = arguments
        .first()
        .is_some_and(|value| value == "--native-child");
    if native_child {
        arguments.remove(0);
    }
    if !native_child {
        #[cfg(all(windows, target_arch = "aarch64"))]
        {
            return supervise::run(&arguments);
        }
        #[cfg(not(all(windows, target_arch = "aarch64")))]
        {
            return Err("The pinned native codec supervisor requires Windows ARM64.".into());
        }
    }
    if arguments.len() != 4 || arguments[0] != "--runtime-directory" || arguments[2] != "--result" {
        return Err("Usage: bonsai-codec-probe --runtime-directory ABSOLUTE_PINNED_DIRECTORY --result ABSOLUTE_JSON_PATH".into());
    }
    let runtime = Path::new(&arguments[1]);
    let result_path = Path::new(&arguments[3]);
    if !runtime.is_absolute()
        || runtime.as_os_str().to_string_lossy().contains('%')
        || runtime
            .file_name()
            .is_none_or(|name| name != RUNTIME_COMMIT)
        || !result_path.is_absolute()
        || result_path.as_os_str().to_string_lossy().contains('%')
    {
        return Err(
            "Use expanded absolute paths and the exact pinned runtime commit directory.".into(),
        );
    }
    let blocks = run_native(runtime)?;
    let result = serde_json::json!({
        "Status": "native-fp32-bit-parity-passed", "SyntheticBlocks": blocks,
        "Fp32Values": blocks * 128, "NativePq2Type": 142, "NativePtq1Type": 143,
        "TraitsBytes": 56, "ToFloatOffset": 40, "NativeRuntimeCommit": RUNTIME_COMMIT,
        "ModelLoaded": false,
        "Scope": "Synthetic native dequantization and exact symbol/scale packing only; activation quantization, logits and performance unqualified."
    });
    let bytes = serde_json::to_vec_pretty(&result).map_err(|error| error.to_string())?;
    let mut output = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(result_path)
        .map_err(|error| error.to_string())?;
    output
        .write_all(&bytes)
        .map_err(|error| error.to_string())?;
    Ok(result)
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
