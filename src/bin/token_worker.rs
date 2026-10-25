//! Explicit-startup, persistent metadata-only JSONL tokenization worker.

#![forbid(unsafe_code)]

use std::ffi::OsString;
use std::io::{self, BufReader, Write};
use std::path::PathBuf;
use std::process::ExitCode;

use bonsai_prism_runtime::special_input::SpecialInputPolicy;
use bonsai_prism_runtime::token_worker::{WorkerConfig, serve};
use bonsai_prism_runtime::tokenization::{
    SpecialTokenInsertion, TextTokenizationPolicy, TokenBudget, TokenizationSession,
};
use bonsai_prism_runtime::{Error, Result};

struct Startup {
    model: PathBuf,
    bytes: u64,
    sha256: String,
    memory_bytes: usize,
    worker: WorkerConfig,
}

const USAGE: &[u8] = b"Usage: bonsai-token-worker --model <absolute-path> --bytes <u64> --sha256 <hex64> --memory-bytes <usize> --context <1..262144> --reserved-output-tokens <usize> --special-input <literal|parse> --insert-special <none|model> --max-frame-bytes <1..1048576>\n\nAll options are required exactly once. Output reservation must be less than context.\nArtifact verification and tokenizer setup precede a flushed readiness JSON record.\nJSONL requests: {\"id\":1,\"op\":\"count\",\"text\":\"complete prompt\"}\nUse op=encode to include exact token IDs. Raw IDs use count_tokens or encode_tokens\nwith token_ids instead of text; no parsing or insertion applies to raw IDs.\nRequest errors are private and recoverable. Every frame requires LF.\nOversized frames or transport I/O errors terminate the worker.\nUse --help alone to show this usage.\n";

fn invalid_options() -> Error {
    io::Error::new(io::ErrorKind::InvalidInput, "invalid token worker options").into()
}

fn number<T: std::str::FromStr>(value: OsString) -> Result<T> {
    let value = value.into_string().map_err(|_| invalid_options())?;
    if value.is_empty() || !value.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(invalid_options());
    }
    value.parse().map_err(|_| invalid_options())
}

// Every option is required exactly once. No text or request data is accepted on
// the command line, and all failures are rendered using a static safe diagnostic.
fn parse(arguments: impl IntoIterator<Item = OsString>) -> Result<Startup> {
    let mut arguments = arguments.into_iter();
    let mut model = None;
    let mut bytes = None;
    let mut sha256 = None;
    let mut memory_bytes = None;
    let mut context = None;
    let mut reserved_output_tokens = None;
    let mut special_input = None;
    let mut insert_special = None;
    let mut max_frame_bytes = None;
    while let Some(key) = arguments.next() {
        let value = arguments.next().ok_or_else(invalid_options)?;
        match key.to_str() {
            Some("--model") if model.is_none() && !value.is_empty() => {
                model = Some(PathBuf::from(value));
            }
            Some("--bytes") if bytes.is_none() => bytes = Some(number::<u64>(value)?),
            Some("--sha256") if sha256.is_none() => {
                let value = value.into_string().map_err(|_| invalid_options())?;
                if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
                    return Err(invalid_options());
                }
                sha256 = Some(value);
            }
            Some("--memory-bytes") if memory_bytes.is_none() => {
                memory_bytes = Some(number::<usize>(value)?);
            }
            Some("--context") if context.is_none() => context = Some(number::<usize>(value)?),
            Some("--reserved-output-tokens") if reserved_output_tokens.is_none() => {
                reserved_output_tokens = Some(number::<usize>(value)?);
            }
            Some("--max-frame-bytes") if max_frame_bytes.is_none() => {
                max_frame_bytes = Some(number::<usize>(value)?);
            }
            Some("--special-input") if special_input.is_none() => {
                special_input = Some(match value.to_str() {
                    Some("literal") => SpecialInputPolicy::Literal,
                    Some("parse") => SpecialInputPolicy::Parse,
                    _ => return Err(invalid_options()),
                });
            }
            Some("--insert-special") if insert_special.is_none() => {
                insert_special = Some(match value.to_str() {
                    Some("none") => SpecialTokenInsertion::None,
                    Some("model") => SpecialTokenInsertion::Model,
                    _ => return Err(invalid_options()),
                });
            }
            _ => return Err(invalid_options()),
        }
    }
    let startup = Startup {
        model: model.ok_or_else(invalid_options)?,
        bytes: bytes
            .filter(|&value| value > 0)
            .ok_or_else(invalid_options)?,
        sha256: sha256.ok_or_else(invalid_options)?,
        memory_bytes: memory_bytes
            .filter(|&value| value > 0)
            .ok_or_else(invalid_options)?,
        worker: WorkerConfig {
            text_policy: TextTokenizationPolicy {
                budget: TokenBudget {
                    context: context.ok_or_else(invalid_options)?,
                    reserved_output_tokens: reserved_output_tokens.ok_or_else(invalid_options)?,
                },
                special_input: special_input.ok_or_else(invalid_options)?,
                insert_special: insert_special.ok_or_else(invalid_options)?,
            },
            max_frame_bytes: max_frame_bytes.ok_or_else(invalid_options)?,
        },
    };
    if !startup.model.is_absolute() {
        return Err(invalid_options());
    }
    startup.worker.validate()?;
    Ok(startup)
}

fn load(
    arguments: impl IntoIterator<Item = OsString>,
) -> Result<(TokenizationSession, WorkerConfig)> {
    let startup = parse(arguments)?;
    let session = TokenizationSession::load(
        &startup.model,
        startup.bytes,
        &startup.sha256,
        startup.memory_bytes,
    )?;
    Ok((session, startup.worker))
}

fn main() -> ExitCode {
    let arguments: Vec<_> = std::env::args_os().skip(1).collect();
    if arguments.len() == 1 && arguments[0] == "--help" {
        return if io::stdout().lock().write_all(USAGE).is_ok() {
            ExitCode::SUCCESS
        } else {
            ExitCode::FAILURE
        };
    }
    let Ok((session, config)) = load(arguments) else {
        let _ = io::stderr()
            .lock()
            .write_all(b"token worker startup failed\n");
        return ExitCode::FAILURE;
    };
    // An explicit small read buffer is independent of the admitted frame bound.
    let mut input = BufReader::with_capacity(8192, io::stdin().lock());
    let mut output = io::stdout().lock();
    if serve(&session, config, &mut input, &mut output).is_ok() {
        ExitCode::SUCCESS
    } else {
        let _ = io::stderr()
            .lock()
            .write_all(b"token worker transport failed\n");
        ExitCode::FAILURE
    }
}
