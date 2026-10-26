//! Process boundary to the source.lock.json-pinned production ggml library.
//!
//! Inputs, reference calculations, tolerances and assertions belong to the
//! calling Rust conformance harness, never to this adapter.

#![deny(warnings, unsafe_code)]
#![deny(unsafe_op_in_unsafe_fn)]

#[allow(
    unsafe_code,
    reason = "The isolated native module implements the pinned ggml C ABI"
)]
mod native;
mod protocol;

use std::env;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Write as _};
use std::path::{Component, Path, PathBuf};
use std::process::ExitCode;

fn normalized(path: &Path) -> PathBuf {
    let mut result = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                result.pop();
            }
            other => result.push(other.as_os_str()),
        }
    }
    result
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    protocol::require(
        cfg!(target_endian = "little"),
        "packed ggml protocol requires a little-endian host",
    )?;
    let arguments: Vec<_> = env::args_os().collect();
    protocol::require(
        arguments.len() == 5 && arguments[1] == "--input" && arguments[3] == "--output",
        "usage: prism-upstream-bridge --input ABSOLUTE_PATH --output ABSOLUTE_PATH",
    )?;
    let input = Path::new(&arguments[2]);
    let output = Path::new(&arguments[4]);
    protocol::require(
        input.is_absolute() && output.is_absolute(),
        "protocol paths must be absolute",
    )?;
    protocol::require(
        normalized(input) != normalized(output),
        "input and output paths must differ",
    )?;
    protocol::require(
        fs::metadata(input)?.len() <= protocol::MAX_FILE_BYTES,
        "request file exceeds size limit",
    )?;
    let input = File::open(input).map_err(|_| "cannot open request file")?;
    let requests = protocol::read_requests(BufReader::new(input))?;
    let output = File::create(output).map_err(|_| "cannot open response file")?;
    let mut output = BufWriter::new(output);
    protocol::write_responses(&mut output, &requests, native::execute)?;
    output.flush()?;
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            let _ = writeln!(std::io::stderr().lock(), "prism-upstream-bridge: {error}");
            ExitCode::FAILURE
        }
    }
}
