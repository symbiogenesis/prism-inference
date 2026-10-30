//! Value-free startup diagnostics for the independently owned private process.

use std::{
    error::Error,
    fmt,
    io::{self, Write},
};

/// Fixed startup boundary, without paths, arguments, credentials or model values.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Stage {
    /// Strict arguments and selected read-only file paths.
    Arguments,
    /// Rejected process environment overrides.
    Environment,
    /// Physical-memory query or startup admission.
    Memory,
    /// Retained private credential reading.
    Key,
    /// Exact model, chat, vocabulary, adapter and graph construction.
    Model,
    /// Async runtime construction.
    Runtime,
    /// Exclusive numeric-loopback binding and listener creation.
    Listener,
    /// Private serving and joined worker lifecycle.
    Backend,
    /// An error outside the classified startup boundaries.
    Unknown,
}

impl Stage {
    const fn name(self) -> &'static str {
        match self {
            Self::Arguments => "arguments",
            Self::Environment => "environment",
            Self::Memory => "memory",
            Self::Key => "key",
            Self::Model => "model",
            Self::Runtime => "runtime",
            Self::Listener => "listener",
            Self::Backend => "backend",
            Self::Unknown => "unknown",
        }
    }
}

#[derive(Debug)]
struct Classification {
    stage: Stage,
    kind: io::ErrorKind,
    os_code: Option<i32>,
}

impl fmt::Display for Classification {
    fn fmt(&self, output: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            output,
            "private-inference-startup-failed stage={} errorKind={:?} osCode=",
            self.stage.name(),
            self.kind
        )?;
        if let Some(code) = self.os_code {
            write!(output, "{code}")
        } else {
            output.write_str("none")
        }
    }
}

impl Error for Classification {}

#[cfg(any(windows, test))]
pub(crate) fn at(stage: Stage, error: &(dyn Error + 'static)) -> io::Error {
    let io_error = error.downcast_ref::<io::Error>();
    let kind = io_error.map_or(io::ErrorKind::Other, io::Error::kind);
    io::Error::new(
        kind,
        Classification {
            stage,
            kind,
            os_code: io_error.and_then(io::Error::raw_os_error),
        },
    )
}

/// Write only the fixed stage, I/O error kind and optional numeric OS code.
///
/// Arbitrary error messages and source chains are never formatted or retained.
/// Unclassified errors receive the fixed `unknown` stage.
///
/// # Errors
/// Returns a failed write to the caller-owned diagnostic stream.
pub fn write_failure(output: &mut impl Write, error: &io::Error) -> io::Result<()> {
    if let Some(classification) = error
        .get_ref()
        .and_then(|error| error.downcast_ref::<Classification>())
    {
        writeln!(output, "{classification}")
    } else {
        let safe = Classification {
            stage: Stage::Unknown,
            kind: error.kind(),
            os_code: error.raw_os_error(),
        };
        writeln!(output, "{safe}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn startup_diagnostics_keep_only_stage_kind_and_numeric_os_code() -> io::Result<()> {
        for (stage, name) in [
            (Stage::Arguments, "arguments"),
            (Stage::Environment, "environment"),
            (Stage::Memory, "memory"),
            (Stage::Key, "key"),
            (Stage::Model, "model"),
            (Stage::Runtime, "runtime"),
            (Stage::Listener, "listener"),
            (Stage::Backend, "backend"),
        ] {
            let private = io::Error::new(
                io::ErrorKind::PermissionDenied,
                "PRIVATE_KEY_AND_PATH\nFORGED stage=ok",
            );
            let mut output = Vec::new();
            write_failure(&mut output, &at(stage, &private))?;
            assert_eq!(output, format!("private-inference-startup-failed stage={name} errorKind=PermissionDenied osCode=none\n").as_bytes());
        }
        let os_error = io::Error::from_raw_os_error(5);
        let mut output = Vec::new();
        write_failure(&mut output, &at(Stage::Key, &os_error))?;
        assert_eq!(
            output,
            format!(
                "private-inference-startup-failed stage=key errorKind={:?} osCode=5\n",
                os_error.kind()
            )
            .as_bytes()
        );
        output.clear();
        write_failure(
            &mut output,
            &io::Error::other("PRIVATE_UNCLASSIFIED_MESSAGE"),
        )?;
        assert_eq!(
            output,
            b"private-inference-startup-failed stage=unknown errorKind=Other osCode=none\n"
        );
        Ok(())
    }
}
