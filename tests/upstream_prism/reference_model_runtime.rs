//! Clean-room safe Rust replacement for `tools/prism/upstream/common/` C++ session tools.
//!
//! No `bonsai_prism_runtime`, `bonsai_gguf`, or native GGML libraries are linked or called.

mod architectures;
mod dspark;
mod forward;
mod gguf_model;
mod llg_vocab;
mod lora_merge;
mod params;
mod sampler;
mod session;

use std::io::{Read, Write};
use std::path::Path;

use crate::Result;

pub(crate) const IDS: &[&str] = &[
    "tests/test-backend-sampler.cpp",
    "tests/test-export-graph-ops.cpp",
    "tests/test-lora-conversion-inference.sh",
    "tests/test-tokenizer-random.py",
    "tests/test-tokenizer-1-spm.cpp",
    "tests/test-tokenizer-1-bpe.cpp",
    "tests/test-dspark-forward.cpp",
    "tests/test-dspark-loop.cpp",
    "tests/test-dspark-real-eval.cpp",
    "tests/test-llama-archs.cpp",
    "tests/test-recurrent-state-rollback.cpp",
    "tests/test-kv-mean-center.cpp",
    "tests/test-autorelease.cpp",
    "tests/test-model-load-cancel.cpp",
    "tests/test-save-load-state.cpp",
    "tests/test-state-restore-fragmented.cpp",
    "tests/test-thread-safety.cpp",
];

pub(crate) fn identity() -> serde_json::Value {
    serde_json::json!({
        "Kind": "independent-rust-model-runtime",
        "SupportedSourceIds": IDS,
        "UpstreamCommit": super::UPSTREAM,
        "Architecture": "clean-room-gguf-forward-dspark-lora-sampler-session-v1",
        "NativeReferenceReplay": false,
        "ProductionRuntimeCalled": false,
        "CompleteSourceGate": false,
        "NativeRetirementQualified": false,
    })
}

#[cfg(test)]
pub(crate) fn run_framed_session<R: Read, W: Write + Send>(
    reader: R,
    writer: W,
    session_dir: &Path,
) -> Result<()> {
    session::run_framed_session_with_env(reader, writer, session_dir, &[])
}

pub(crate) fn run_framed_session_with_env<R: Read, W: Write + Send>(
    reader: R,
    writer: W,
    session_dir: &Path,
    initial_env: &[(String, String)],
) -> Result<()> {
    session::run_framed_session_with_env(reader, writer, session_dir, initial_env)
}

pub(crate) fn export_lora_in_process(
    base: &Path,
    adapter: &Path,
    output: &Path,
    directory: &Path,
) -> Result<serde_json::Value> {
    lora_merge::export_lora_in_process(base, adapter, output, directory)
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use serde_json::{Value, json};

    use super::*;

    #[test]
    fn framed_session_runs_basic_lifecycle_and_shutdown() -> Result<()> {
        let dir = std::env::temp_dir().join("prism-ref-model-runtime-basic");
        std::fs::create_dir_all(&dir)?;
        let input = concat!(
            r#"{"Id":1,"Worker":0,"Operation":{"Op":"devices"}}"#,
            "\n",
            r#"{"Id":2,"Worker":0,"Operation":{"Op":"architecture-inventory"}}"#,
            "\n",
            r#"{"Id":3,"Shutdown":true}"#,
            "\n",
        );
        let mut out = Vec::new();
        run_framed_session(Cursor::new(input.as_bytes()), &mut out, &dir)?;
        let text = String::from_utf8(out)?;
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 3);
        let first: Value = serde_json::from_str(lines[0])?;
        assert_eq!(first["Result"]["Devices"][0]["CpuBuffer"], json!(true));
        let second: Value = serde_json::from_str(lines[1])?;
        assert!(
            !second["Result"]["Architectures"]
                .as_array()
                .is_none_or(Vec::is_empty)
        );
        let third: Value = serde_json::from_str(lines[2])?;
        assert_eq!(third, json!({ "Shutdown": true }));
        let _ = std::fs::remove_dir_all(&dir);
        Ok(())
    }

    #[test]
    fn process_environment_remains_isolated_in_session() -> Result<()> {
        let dir = std::env::temp_dir().join("prism-ref-model-runtime-env");
        std::fs::create_dir_all(&dir)?;
        let input = concat!(
            r#"{"Id":1,"Worker":0,"Operation":{"Op":"process-environment","Changes":{"LLAMA_DSPARK_SNR_TWO_ROWS":true},"Save":true}}"#,
            "\n",
            r#"{"Id":2,"Worker":0,"Operation":{"Op":"process-environment-restore","Environment":{"LLAMA_DSPARK_SNR_TWO_ROWS":null}}}"#,
            "\n",
            r#"{"Id":3,"Shutdown":true}"#,
            "\n",
        );
        let mut out = Vec::new();
        run_framed_session(Cursor::new(input.as_bytes()), &mut out, &dir)?;
        assert!(std::env::var_os("LLAMA_DSPARK_SNR_TWO_ROWS").is_none());
        let _ = std::fs::remove_dir_all(&dir);
        Ok(())
    }
}
