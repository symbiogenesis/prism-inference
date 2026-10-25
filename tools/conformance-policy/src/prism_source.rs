//! Deterministic ordinary Prism source partitioning and revision-bound fan-in.
//!
//! Artifacts accept UTF-8 and BOM-marked UTF-8, UTF-16 or UTF-32 JSON.

use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{File, OpenOptions};
use std::io::Read as _;
use std::path::{Component, Path, PathBuf};

use serde_json::{Map, Value, json};
use sha2::{Digest as _, Sha256};

use crate::Result;

const BACKEND: &str = "tests/test-backend-ops.cpp";
const RECEIPT_LIMIT: u64 = 1024 * 1024;
const REPORT_LIMIT: u64 = 128 * 1024 * 1024;

fn require(condition: bool, message: &str) -> Result<()> {
    if condition {
        Ok(())
    } else {
        Err(message.into())
    }
}

fn object(value: &Value) -> Result<&Map<String, Value>> {
    value
        .as_object()
        .ok_or_else(|| "Prism source record must be an object.".into())
}

fn text<'a>(value: &'a Value, name: &str) -> Result<&'a str> {
    value
        .get(name)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("Prism source field {name} must be text."))
}

fn array<'a>(value: &'a Value, name: &str) -> Result<&'a Vec<Value>> {
    value
        .get(name)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("Prism source field {name} must be an array."))
}

fn digest_identity(value: &str, length: usize) -> Result<()> {
    require(
        value.len() == length
            && value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
        "Prism source revision or digest must be lowercase hexadecimal.",
    )
}

fn exact_fields(value: &Value, fields: &[&str]) -> Result<()> {
    let object = object(value)?;
    require(
        object.len() == fields.len() && fields.iter().all(|field| object.contains_key(*field)),
        "Prism source request fields differ from its operation.",
    )
}

fn partition(id: &str) -> u64 {
    match id {
        "tests/test-tokenizer-random.py" => 0,
        "tests/test-thread-safety.cpp" => 1,
        "tests/test-quantize-stats.cpp" | "tests/test-grammar-llguidance.cpp" => 2,
        _ => 3,
    }
}

fn select(request: &Value) -> Result<Value> {
    exact_fields(request, &["Operation", "Cases", "Index"])?;
    let index = request["Index"]
        .as_u64()
        .filter(|index| *index < 4)
        .ok_or("Prism source partition must be an integer from zero through three.")?;
    let mut ids = BTreeSet::new();
    let mut ordinals = BTreeSet::new();
    let mut selected = Vec::new();
    for case in array(request, "Cases")? {
        exact_fields(case, &["Id", "Ordinal"])?;
        let id = text(case, "Id")?;
        let ordinal = case["Ordinal"]
            .as_u64()
            .filter(|ordinal| i32::try_from(*ordinal).is_ok())
            .ok_or("Prism source ordinal must be a nonnegative 32-bit integer.")?;
        require(
            !id.trim().is_empty() && ids.insert(id) && ordinals.insert(ordinal),
            "Prism source IDs and ordinals must be nonempty and unique.",
        )?;
        if partition(id) == index {
            selected.push(ordinal);
        }
    }
    selected.sort_unstable();
    Ok(json!({"Ordinals": selected}))
}

fn expanded_path(path: &Path) -> Result<PathBuf> {
    let spelling = path.to_str().ok_or("Prism source path must be Unicode.")?;
    let namespace = spelling.replace('/', "\\");
    let positions: Vec<_> = spelling
        .match_indices('%')
        .map(|(index, _)| index)
        .collect();
    require(
        path.is_absolute()
            && !positions.windows(2).any(|pair| pair[1] > pair[0] + 1)
            && !namespace.starts_with("\\\\?\\")
            && !namespace.starts_with("\\\\.\\"),
        "Use an expanded absolute Prism source path without device syntax.",
    )?;
    let full = std::path::absolute(path).map_err(|error| error.to_string())?;
    require(
        !full.components().any(
            |part| matches!(part, Component::Normal(name) if name.to_string_lossy().contains(':')),
        ),
        "Alternate data stream paths are unsupported.",
    )?;
    Ok(full)
}

fn redirect(metadata: &std::fs::Metadata) -> bool {
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt as _;
        metadata.file_attributes() & 0x400 != 0
    }
    #[cfg(not(windows))]
    {
        metadata.file_type().is_symlink()
    }
}

fn external_path(path: &Path) -> Result<PathBuf> {
    let full = expanded_path(path)?;
    for ancestor in full.ancestors() {
        let metadata = ancestor
            .symlink_metadata()
            .map_err(|error| error.to_string())?;
        require(
            !redirect(&metadata),
            "Reparse-point Prism source storage is unsupported.",
        )?;
        if metadata.is_dir() {
            match ancestor.join(".git").symlink_metadata() {
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => return Err(error.to_string()),
                Ok(_) => {
                    return Err("Prism source storage must be outside Git repositories.".into());
                }
            }
        }
    }
    Ok(full)
}

// Keep every input and, on Windows, its ancestors stable until fan-in completes.
// Receipt hashes bind exactly the bytes parsed from these retained read handles.
struct Inputs {
    handles: Vec<File>,
}

impl Inputs {
    const fn new() -> Self {
        Self {
            handles: Vec::new(),
        }
    }

    fn read(&mut self, path: &Path, maximum: u64) -> Result<(Value, String)> {
        let full = external_path(path)?;
        #[cfg(windows)]
        {
            use std::os::windows::fs::OpenOptionsExt as _;
            for ancestor in full.ancestors().skip(1) {
                let directory = OpenOptions::new()
                    .read(true)
                    .access_mode(0x80)
                    .share_mode(1)
                    .custom_flags(0x0220_0000)
                    .open(ancestor)
                    .map_err(|error| {
                        format!(
                            "Cannot retain Prism source ancestor {}: {error}",
                            ancestor.display()
                        )
                    })?;
                let metadata = directory.metadata().map_err(|error| error.to_string())?;
                require(
                    metadata.is_dir() && !redirect(&metadata),
                    "Prism source ancestor changed.",
                )?;
                self.handles.push(directory);
            }
        }
        let mut options = OpenOptions::new();
        options.read(true);
        #[cfg(windows)]
        {
            use std::os::windows::fs::OpenOptionsExt as _;
            // FILE_SHARE_READ only; OPEN_REPARSE_POINT rejects a leaf redirect.
            options.share_mode(1).custom_flags(0x0020_0000);
        }
        let mut file = options.open(&full).map_err(|error| error.to_string())?;
        let metadata = file.metadata().map_err(|error| error.to_string())?;
        require(
            metadata.is_file() && !redirect(&metadata),
            "Prism source artifact is not an ordinary file.",
        )?;
        require(
            metadata.len() <= maximum,
            "Prism source artifact exceeds its byte bound.",
        )?;
        let mut bytes = Vec::new();
        file.by_ref()
            .take(maximum + 1)
            .read_to_end(&mut bytes)
            .map_err(|error| error.to_string())?;
        require(
            u64::try_from(bytes.len()).map_err(|error| error.to_string())? <= maximum,
            "Prism source artifact exceeds its byte bound.",
        )?;
        let digest: String = Sha256::digest(&bytes)
            .iter()
            .flat_map(|byte| {
                const HEX: &[u8; 16] = b"0123456789abcdef";
                [
                    char::from(HEX[usize::from(byte >> 4)]),
                    char::from(HEX[usize::from(byte & 15)]),
                ]
            })
            .collect();
        let text = decode_text(&bytes)?;
        let value: Value = serde_json::from_str(&text).map_err(|error| error.to_string())?;
        bounded_depth(&value, 100)?;
        object(&value)?;
        self.handles.push(file);
        Ok((value, digest))
    }
}

fn decode_text(bytes: &[u8]) -> Result<Cow<'_, str>> {
    // Get-Content recognizes these BOMs. Keep valid legacy encodings while
    // rejecting malformed text rather than replacing evidence bytes silently.
    if bytes.starts_with(&[0xff, 0xfe, 0, 0]) || bytes.starts_with(&[0, 0, 0xfe, 0xff]) {
        let little = bytes[0] == 0xff;
        let (units, trailing) = bytes[4..].as_chunks::<4>();
        require(
            trailing.is_empty(),
            "Truncated UTF-32 Prism source artifact.",
        )?;
        let text: Result<String> = units
            .iter()
            .map(|unit| {
                char::from_u32(if little {
                    u32::from_le_bytes(*unit)
                } else {
                    u32::from_be_bytes(*unit)
                })
                .ok_or_else(|| "Invalid UTF-32 Prism source artifact.".into())
            })
            .collect();
        return Ok(Cow::Owned(text?));
    }
    if bytes.starts_with(&[0xff, 0xfe]) || bytes.starts_with(&[0xfe, 0xff]) {
        let little = bytes[0] == 0xff;
        let (units, trailing) = bytes[2..].as_chunks::<2>();
        require(
            trailing.is_empty(),
            "Truncated UTF-16 Prism source artifact.",
        )?;
        let text: Result<String> = char::decode_utf16(units.iter().map(|unit| {
            if little {
                u16::from_le_bytes(*unit)
            } else {
                u16::from_be_bytes(*unit)
            }
        }))
        .map(|character| character.map_err(|error| error.to_string()))
        .collect();
        return Ok(Cow::Owned(text?));
    }
    let bytes = bytes.strip_prefix(&[0xef, 0xbb, 0xbf]).unwrap_or(bytes);
    Ok(Cow::Borrowed(
        std::str::from_utf8(bytes).map_err(|error| error.to_string())?,
    ))
}

fn bounded_depth(value: &Value, remaining: usize) -> Result<()> {
    match value {
        Value::Array(items) => {
            require(
                remaining > 0,
                "Prism source artifact exceeds its depth bound.",
            )?;
            for child in items {
                bounded_depth(child, remaining - 1)?;
            }
        }
        Value::Object(fields) => {
            require(
                remaining > 0,
                "Prism source artifact exceeds its depth bound.",
            )?;
            for child in fields.values() {
                bounded_depth(child, remaining - 1)?;
            }
        }
        _ => {}
    }
    Ok(())
}

fn ordinary_ids(catalog: &Value) -> Result<Vec<&str>> {
    object(catalog)?;
    digest_identity(text(catalog, "UpstreamCommit")?, 40)?;
    digest_identity(text(catalog, "ArchiveSha256")?, 64)?;
    let mut seen = BTreeSet::new();
    let mut ids = Vec::new();
    for value in array(catalog, "AdapterIds")? {
        let id = value
            .as_str()
            .filter(|id| !id.trim().is_empty())
            .ok_or("Prism adapter ID must be nonempty text.")?;
        require(seen.insert(id), "Prism adapter IDs must be unique.")?;
        if id != BACKEND {
            ids.push(id);
        }
    }
    Ok(ids)
}

fn common_bridge(receipt: &Value) -> Result<&str> {
    let mut common = None;
    for build in array(receipt, "NativeBuilds")? {
        object(build)?;
        if text(build, "Family")? == "common" {
            require(
                common.is_none(),
                "Prism source receipt repeats the common native build.",
            )?;
            let digest = text(build, "ExecutableSha256")?;
            digest_identity(digest, 64)?;
            common = Some(digest);
        }
    }
    common.ok_or_else(|| "Prism source receipt lacks its common native build.".into())
}

fn validate_receipt<'a>(receipt: &'a Value, request: &Value, index: u64) -> Result<&'a str> {
    require(
        receipt["SchemaVersion"].as_u64() == Some(1)
            && text(receipt, "ProjectRevision")? == text(request, "ProjectRevision")?
            && text(receipt, "SourceCommit")? == text(&request["Catalog"], "UpstreamCommit")?
            && text(receipt, "SourceArchiveSha256")? == text(&request["Catalog"], "ArchiveSha256")?
            && text(receipt, "Status")? == "awaiting-backend"
            && text(receipt, "Selection")? == "All"
            && receipt["SourcePartition"].as_u64() == Some(index)
            && receipt["SourcePartitionCount"].as_u64() == Some(4)
            && text(receipt, "NativePlatform")? == "ARM64"
            && text(receipt, "Architecture")? == "Arm64"
            && text(receipt, "DeferredSource")? == BACKEND,
        "Ordinary Prism source partition provenance differs.",
    )?;
    let harness = text(receipt, "HarnessSha256")?;
    digest_identity(harness, 64)?;
    let bridge = common_bridge(receipt)?;
    array(receipt, "CaseRuns")?;
    if index == 0 {
        require(
            harness == text(request, "HarnessSha256")?
                && bridge == text(request, "CommonBridgeSha256")?,
            "Ordinary partition zero differs from the backend tools.",
        )?;
    }
    Ok(bridge)
}

fn merge(request: &Value) -> Result<Value> {
    exact_fields(
        request,
        &[
            "Operation",
            "InputDirectory",
            "Catalog",
            "ProjectRevision",
            "HarnessSha256",
            "CommonBridgeSha256",
        ],
    )?;
    digest_identity(text(request, "ProjectRevision")?, 40)?;
    digest_identity(text(request, "HarnessSha256")?, 64)?;
    digest_identity(text(request, "CommonBridgeSha256")?, 64)?;
    let root = external_path(Path::new(text(request, "InputDirectory")?))?;
    let ids = ordinary_ids(&request["Catalog"])?;
    let upstream = text(&request["Catalog"], "UpstreamCommit")?;
    let mut inputs = Inputs::new();
    let mut combined = Value::Null;
    let mut receipts = Vec::new();
    let mut case_runs = Vec::new();
    let mut results = [BTreeMap::new(), BTreeMap::new()];
    for index in 0..4 {
        let directory = root.join(format!("prism-ordinary-results-{index}"));
        let (receipt, digest) = inputs.read(&directory.join("receipt.json"), RECEIPT_LIMIT)?;
        let bridge = validate_receipt(&receipt, request, index)?;
        let expected: BTreeSet<_> = ids
            .iter()
            .copied()
            .filter(|id| partition(id) == index)
            .collect();
        for (target, result) in ["cpp", "rust"].into_iter().zip(&mut results) {
            let (report, _) = inputs.read(
                &directory.join(format!("{target}-results.json")),
                REPORT_LIMIT,
            )?;
            let cases = array(&report, "Cases")?;
            require(
                report["SchemaVersion"].as_u64() == Some(1)
                    && text(&report, "UpstreamCommit")? == upstream
                    && cases.len() == expected.len(),
                "Ordinary Prism source result identity or count differs.",
            )?;
            let mut remaining = expected.clone();
            for case in cases {
                object(case)?;
                let id = text(case, "Id")?;
                require(
                    remaining.remove(id) && result.insert(id.to_owned(), case.clone()).is_none(),
                    "Ordinary Prism source partition contains a duplicate or misplaced source.",
                )?;
            }
        }
        receipts.push(json!({
            "Index":index, "SourceCount":expected.len(), "HarnessSha256":text(&receipt,"HarnessSha256")?,
            "CommonBridgeSha256":bridge, "ReceiptSha256":digest, "CaseRuns":receipt["CaseRuns"]
        }));
        case_runs.extend(array(&receipt, "CaseRuns")?.iter().cloned());
        if index == 0 {
            combined = receipt;
        }
    }
    combined["SourcePartition"] = json!(-1);
    combined["SourcePartitions"] = Value::Array(receipts);
    combined["CaseRuns"] = Value::Array(case_runs);
    let mut merged = json!({"Receipt":combined});
    for (target, mut result) in ["cpp", "rust"].into_iter().zip(results) {
        let cases: Result<Vec<_>> = ids
            .iter()
            .map(|id| {
                result
                    .remove(*id)
                    .ok_or_else(|| "Missing ordinary Prism source result.".into())
            })
            .collect();
        merged[target] = json!({"SchemaVersion":1, "UpstreamCommit":upstream, "Cases":cases?});
    }
    Ok(merged)
}

/// Select original source ordinals or merge all four ordinary source partitions.
///
/// # Errors
/// Rejects malformed identities, unsafe or oversized artifacts, foreign receipts,
/// and missing, duplicate or misplaced source results.
#[allow(
    clippy::needless_pass_by_value,
    reason = "Maintenance command dispatch transfers ownership through the shared command API."
)]
pub fn run(request: Value) -> Result<Value> {
    object(&request)?;
    match text(&request, "Operation")? {
        "partition" => select(&request),
        "merge" => merge(&request),
        _ => Err("Unknown Prism source operation.".into()),
    }
}

#[cfg(test)]
#[path = "prism_source/tests.rs"]
mod tests;
