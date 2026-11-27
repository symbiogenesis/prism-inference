//! Bounded source sanity against Git's immutable outgoing ref tips.
//!
//! Adapted from Runner's native push guard; compilation and complete tests remain
//! the responsibility of Prism's hosted validation lanes.
#![forbid(unsafe_code)]

use std::ffi::OsString;
use std::io::{Read as _, Write as _};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

#[cfg(windows)]
mod windows;

type Result<T> = std::result::Result<T, String>;

const USAGE: &str = "prism-source-sanity --root PATH --policy-root PATH --bootstrap-milliseconds N [--check-commit OID]; otherwise Git pre-push updates on stdin";
const MAX_UPDATES: u64 = 65_536;

#[derive(Debug, PartialEq, Eq)]
struct Update {
    tip: String,
    base: String,
}

fn oid(value: &str) -> bool {
    matches!(value.len(), 40 | 64) && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn updates(text: &str) -> Result<Vec<Update>> {
    let mut result = Vec::new();
    for line in text.lines().filter(|line| !line.trim().is_empty()) {
        let fields: Vec<_> = line.split_whitespace().collect();
        if fields.len() != 4
            || !oid(fields[1])
            || !oid(fields[3])
            || fields[1].len() != fields[3].len()
        {
            return Err("Push blocked: malformed Git pre-push ref update.".into());
        }
        if !fields[1].bytes().all(|byte| byte == b'0') {
            result.push(Update {
                tip: fields[1].to_ascii_lowercase(),
                base: fields[3].to_ascii_lowercase(),
            });
        }
    }
    Ok(result)
}

fn read_updates(input: impl std::io::Read) -> Result<Vec<Update>> {
    let mut text = String::new();
    input
        .take(MAX_UPDATES + 1)
        .read_to_string(&mut text)
        .map_err(|error| error.to_string())?;
    if u64::try_from(text.len()).map_err(|error| error.to_string())? > MAX_UPDATES {
        return Err("Push blocked: ref-update input exceeds 64 KiB.".into());
    }
    updates(&text)
}

fn rust_selected(paths: &[String]) -> bool {
    paths.iter().any(|path| {
        let path = Path::new(path);
        path.extension()
            .is_some_and(|extension| extension.eq_ignore_ascii_case("rs"))
            || matches!(
                path.file_name().and_then(|name| name.to_str()),
                Some(
                    "Cargo.toml"
                        | "Cargo.lock"
                        | "rustfmt.toml"
                        | ".rustfmt.toml"
                        | "rust-toolchain.toml"
                )
            )
            || path.starts_with(".cargo")
    })
}

const TOOLCHAIN_ERROR: &str =
    "Push formatting requires the outgoing source's dated nightly compiler pin.";

fn strip_comment(line: &str) -> &str {
    let mut quote: Option<u8> = None;
    for (index, byte) in line.bytes().enumerate() {
        match (quote, byte) {
            (None, b'\'' | b'"') => quote = Some(byte),
            (None, b'#') => return &line[..index],
            (Some(open), current) if current == open => quote = None,
            _ => {}
        }
    }
    line
}

fn parse_quoted_value(value: &str) -> Result<&str> {
    let bytes = value.as_bytes();
    let Some(&quote) = bytes.first().filter(|&&byte| matches!(byte, b'\'' | b'"')) else {
        return Err(TOOLCHAIN_ERROR.into());
    };
    let rest = &value[1..];
    let Some(end) = rest.find(char::from(quote)) else {
        return Err(TOOLCHAIN_ERROR.into());
    };
    let inner = &rest[..end];
    if quote == b'"' && inner.contains('\\') {
        return Err(TOOLCHAIN_ERROR.into());
    }
    if !rest[end + 1..].trim().is_empty() {
        return Err(TOOLCHAIN_ERROR.into());
    }
    Ok(inner)
}

fn toolchain(text: &str) -> Result<String> {
    let text = text.strip_prefix('\u{feff}').unwrap_or(text);
    let mut section: Option<&str> = None;
    let mut seen_toolchain = false;
    let mut channel: Option<&str> = None;

    for raw_line in text.lines() {
        let line = strip_comment(raw_line).trim();
        if line.is_empty() {
            continue;
        }
        if let Some(after_open) = line.strip_prefix('[') {
            let Some(name) = after_open.strip_suffix(']') else {
                return Err(TOOLCHAIN_ERROR.into());
            };
            let name = name.trim();
            if name.is_empty() || name.contains(['[', ']']) {
                return Err(TOOLCHAIN_ERROR.into());
            }
            if name == "toolchain" {
                if seen_toolchain {
                    return Err(TOOLCHAIN_ERROR.into());
                }
                seen_toolchain = true;
            }
            section = Some(name);
            continue;
        }
        let Some(current_section) = section else {
            return Err(TOOLCHAIN_ERROR.into());
        };
        let Some((key, value)) = line.split_once('=') else {
            return Err(TOOLCHAIN_ERROR.into());
        };
        let key = key.trim();
        let value = value.trim();
        if key.is_empty() || value.is_empty() {
            return Err(TOOLCHAIN_ERROR.into());
        }
        if current_section == "toolchain" && key == "channel" {
            if channel.is_some() {
                return Err(TOOLCHAIN_ERROR.into());
            }
            channel = Some(parse_quoted_value(value)?);
        }
    }

    let channel = channel
        .filter(|channel| {
            channel.len() == 18
                && channel.starts_with("nightly-")
                && channel.bytes().enumerate().skip(8).all(|(index, byte)| {
                    if matches!(index, 12 | 15) {
                        byte == b'-'
                    } else {
                        byte.is_ascii_digit()
                    }
                })
        })
        .ok_or(TOOLCHAIN_ERROR)?;
    Ok(format!("{channel}-aarch64-pc-windows-msvc"))
}

fn canonical_path(path: impl AsRef<Path>) -> Result<PathBuf> {
    let path = std::fs::canonicalize(path).map_err(|error| error.to_string())?;
    #[cfg(windows)]
    {
        let text = path.to_str().ok_or("Push sanity paths must be UTF-8.")?;
        if let Some(suffix) = text.strip_prefix("\\\\?\\UNC\\") {
            return Ok(PathBuf::from(format!("\\\\{suffix}")));
        }
        if let Some(suffix) = text.strip_prefix("\\\\?\\") {
            return Ok(PathBuf::from(suffix));
        }
    }
    Ok(path)
}

fn run(arguments: &[OsString]) -> Result<String> {
    if !matches!(arguments.len(), 6 | 8)
        || arguments[0] != "--root"
        || arguments[2] != "--policy-root"
        || arguments[4] != "--bootstrap-milliseconds"
        || (arguments.len() == 8 && arguments[6] != "--check-commit")
    {
        return Err(USAGE.into());
    }
    let root = canonical_path(&arguments[1])?;
    let policy = canonical_path(&arguments[3])?;
    let bootstrap = arguments[5]
        .to_str()
        .ok_or(USAGE)?
        .parse::<u64>()
        .map_err(|_| USAGE)?;
    let updates = if arguments.len() == 8 {
        let tip = arguments[7].to_str().filter(|tip| oid(tip)).ok_or(USAGE)?;
        vec![Update {
            tip: tip.to_ascii_lowercase(),
            base: "0".repeat(tip.len()),
        }]
    } else {
        read_updates(std::io::stdin().lock())?
    };
    if updates.is_empty() {
        return Ok("No outgoing source tips (empty input or deletions only).".into());
    }
    #[cfg(windows)]
    return windows::check(&root, &policy, &updates, bootstrap);
    #[cfg(not(windows))]
    {
        let _ = (
            root,
            policy,
            updates,
            bootstrap,
            rust_selected(&[]),
            toolchain(""),
        );
        Err("Push sanity requires native Windows ARM64 and pinned Rust/PowerShell tools.".into())
    }
}

fn main() -> ExitCode {
    match run(&std::env::args_os().skip(1).collect::<Vec<_>>()) {
        Ok(message) => {
            if writeln!(std::io::stdout().lock(), "{message}").is_err() {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_non_head_branches_tags_and_deletions() -> Result<()> {
        for length in [40, 64] {
            let first = "a".repeat(length);
            let second = "b".repeat(length);
            let zero = "0".repeat(length);
            let input = format!(
                "refs/heads/other {first} refs/heads/main {second}\nrefs/tags/v1 {second} refs/tags/v1 {zero}\n(delete) {zero} refs/heads/old {first}\n"
            );
            assert_eq!(
                updates(&input)?,
                [
                    Update {
                        tip: first,
                        base: second.clone()
                    },
                    Update {
                        tip: second,
                        base: zero
                    },
                ]
            );
        }
        Ok(())
    }

    #[test]
    fn rejects_malformed_and_oversized_input() {
        for input in [
            "refs/heads/main HEAD refs/heads/main HEAD".to_owned(),
            "only three fields".to_owned(),
            format!("branch {} branch {}", "a".repeat(40), "b".repeat(64)),
        ] {
            assert!(updates(&input).is_err());
        }
        assert!(read_updates(&vec![b' '; 65_537][..]).is_err());
        assert!(read_updates(&[0xff][..]).is_err());
    }

    #[test]
    fn formats_changes_to_rust_inputs_and_nested_configuration() {
        for path in [
            "src/lib.rs",
            "Cargo.lock",
            "tools/x/Cargo.toml",
            ".cargo/config.toml",
            "tools/x/.rustfmt.toml",
            "rust-toolchain.toml",
        ] {
            assert!(rust_selected(&[path.into()]), "{path}");
        }
        assert!(!rust_selected(&[
            "README.md".into(),
            "docs/rustfmt.toml.example".into()
        ]));
    }

    #[test]
    fn requires_an_explicit_dated_outgoing_compiler() -> Result<()> {
        assert_eq!(
            toolchain("[toolchain]\nchannel = 'nightly-2026-11-09'\n")?,
            "nightly-2026-11-09-aarch64-pc-windows-msvc"
        );
        assert_eq!(
            toolchain(
                "\u{feff}# pin comment\n[other]\nchannel = 'nightly-2026-01-01'\n[toolchain]\nchannel = \"nightly-2026-11-09\" # inline comment\n"
            )?,
            "nightly-2026-11-09-aarch64-pc-windows-msvc"
        );
        for channel in [
            "nightly",
            "stable",
            "nightly-2026-11-09-aarch64-pc-windows-msvc",
        ] {
            assert!(toolchain(&format!("[toolchain]\nchannel='{channel}'\n")).is_err());
        }
        for invalid in [
            "channel = 'nightly-2026-11-09'\n",
            "[toolchain]\nchannel = 'nightly-2026-11-09'\nchannel = 'nightly-2026-11-09'\n",
            "[toolchain]\nchannel = 'nightly-2026-11-09'\n[toolchain]\n",
            "[toolchain\nchannel = 'nightly-2026-11-09'\n",
            "[toolchain]\nchannel = 'nightly-2026-11-09' extra\n",
            "[toolchain]\nchannel = \"nightly-2026\\u002d11-09\"\n",
            "[other]\nchannel = 'nightly-2026-11-09'\n",
        ] {
            assert!(toolchain(invalid).is_err(), "{invalid}");
        }
        Ok(())
    }
}
