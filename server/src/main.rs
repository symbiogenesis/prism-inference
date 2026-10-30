//! Job-owned private HTTP inference executable; no frontend listener is exposed.
#![forbid(unsafe_code)]
use std::{io::Write, process::ExitCode};

fn main() -> ExitCode {
    let arguments: Vec<_> = std::env::args_os().skip(1).collect();
    if arguments.len() == 1 && arguments[0] == "--help" {
        return if std::io::stdout().write_all(b"prism-inference-server --native-engine-child --model <absolute-pinned-gguf> --port <1024..65535> --key-file <absolute-private-key> --seed <u32> [--row-workers <1..16>] [--use-repacked-ptq1] [--abliterate-lora <absolute-approved-gguf>]\nPrivate HTTP backend for an owning process/Job launcher. Default: Swift Bonsai PQ2 without an adapter.\n").is_ok() { ExitCode::SUCCESS } else { ExitCode::FAILURE };
    }
    #[cfg(windows)]
    if prism_inference_server::native_child::run(arguments).is_ok() {
        return ExitCode::SUCCESS;
    }
    let _written = std::io::stderr()
        .write_all(b"Private inference server failed; admission remains closed.\n");
    ExitCode::FAILURE
}
