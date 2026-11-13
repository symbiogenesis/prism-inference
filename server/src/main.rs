//! Job-owned private HTTP inference executable; no frontend listener is exposed.
#![forbid(unsafe_code)]
use std::{io::Write, process::ExitCode};

fn main() -> ExitCode {
    let arguments: Vec<_> = std::env::args_os().skip(1).collect();
    if arguments.len() == 1 && arguments[0] == "--help" {
        return if std::io::stdout().write_all(b"prism-inference-server --native-engine-child --model <absolute-pinned-gguf> --port <1024..65535> --key-file <absolute-private-key> --seed <u32> [--row-workers <1..16>] [--use-repacked-ptq1] [--abliterate-lora <absolute-approved-gguf>] [--kv-cache <f16|rsq4|rsq4vf|rsq4k3v>] [--context-tokens <4352..262144> --memory-limit-bytes <u64>] [--cpu-prefill-tokens <8|16|32> --cpu-prefill-scratch-bytes <usize>] [--cpu-speculative-tokens <1..8> --cpu-verification-scratch-bytes <usize> --cpu-speculative-checkpoint-bytes <usize>]\nPrivate HTTP backend for an owning process/Job launcher. Development default: Swift Bonsai PQ2 without an adapter and rsq4 K4/V4 KV. Use --kv-cache f16 for explicit comparison/recovery; no automatic fallback. Context is page-aligned and memory-admitted; selected-cache ceilings are f16 62720, rsq4 262144, rsq4vf 241664 and rsq4k3v 262144 tokens. rsq4vf is an explicit fitted-value scale candidate. CPU tiles require explicit KV precision and paired context/memory including their additional scratch reserve; they disable automatic NPU placement. Sampled suffix speculation additionally requires tiles and separate logits/checkpoint reserves. Full-model quality and device qualification remain pending.\n").is_ok() { ExitCode::SUCCESS } else { ExitCode::FAILURE };
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
