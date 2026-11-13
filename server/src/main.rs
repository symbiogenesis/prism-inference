//! Job-owned private HTTP inference executable; no frontend listener is exposed.
#![forbid(unsafe_code)]
use std::{io::Write, process::ExitCode};

fn main() -> ExitCode {
    let arguments: Vec<_> = std::env::args_os().skip(1).collect();
    if arguments.len() == 1 && arguments[0] == "--help" {
        return if std::io::stdout().write_all(b"prism-inference-server --native-engine-child --model <absolute-pinned-gguf> --port <1024..65535> --key-file <absolute-private-key> --seed <u32> [--row-workers <1..16>] [--use-repacked-ptq1] [--abliterate-lora <absolute-approved-gguf>] [--kv-cache <f16|rsq4|rsq4k3v>] [--context-tokens <4352..62720> --memory-limit-bytes <u64>] [--cpu-prefill-tokens <8|16|32> --cpu-prefill-scratch-bytes <usize>]\nPrivate HTTP backend for an owning process/Job launcher. Development default: Swift Bonsai PQ2 without an adapter and rsq4 K4/V4 KV. Use --kv-cache f16 for explicit comparison/recovery; no automatic fallback. CPU tiled prefill requires explicit FP16 and paired context/memory including its additional scratch reserve; it disables automatic NPU placement. Full-model quality and device qualification remain pending.\n").is_ok() { ExitCode::SUCCESS } else { ExitCode::FAILURE };
    }
    #[cfg(windows)]
    let result = prism_inference_server::native_child::run(arguments);
    #[cfg(not(windows))]
    let result = Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "unsupported platform",
    ));
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            let _written = prism_inference_server::startup::write_failure(
                &mut std::io::stderr().lock(),
                &error,
            );
            ExitCode::FAILURE
        }
    }
}
