//! Safe Rust replacement for the upstream test-only GGUF header helper.
//!
//! Header parsing, filename selection, progressive retrieval, cache handling and
//! shard merging are target-independent suite code. They are never delegated to
//! C++ or counted as production engine coverage. Input providers supply real,
//! verified GGUF prefix bytes, independently of quantization snapshot schemas.

use super::engine::Engine;
use super::llama_grammar::{array, exact_fields, field, uint};
use super::{Result, invalid};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};

const MAX_HEADER: usize = 64 * 1024 * 1024;
const MAX_ENTRIES: usize = 1_000_000;

/// One raw metadata value; duplicate keys and numeric bit patterns are retained.
#[derive(Clone)]
pub(super) struct Entry {
    /// Raw key bytes, as in the source's `std::string`.
    pub(super) key: Vec<u8>,
    /// GGUF value-type enum.
    pub(super) kind: i32,
    /// Exact serialized value bytes, excluding its type tag.
    pub(super) payload: Vec<u8>,
}
/// One supplied tensor descriptor, retaining all four padded dimensions.
#[derive(Clone)]
pub(super) struct Tensor {
    /// Raw name bytes.
    pub(super) name: Vec<u8>,
    /// Original GGML type enum.
    pub(super) kind: i32,
    /// Source dimensions, including unused dimensions initialized to one.
    pub(super) dimensions: [i64; 4],
    /// Serialized dimension count, which may be zero.
    pub(super) n_dims: u32,
    /// Serialized data offset; header-only parsing does not read tensor values.
    pub(super) offset: u64,
}
/// Selected metadata and raw first-shard KV entries, plus ordered tensor entries.
#[derive(Clone, Default)]
pub(super) struct Model {
    /// GGUF version, either two or three.
    pub(super) version: u32,
    /// Raw architecture text.
    pub(super) architecture: Vec<u8>,
    /// Embedding length.
    pub(super) n_embd: u32,
    /// Feed-forward length, using the first value when array-valued.
    pub(super) n_ff: u32,
    /// Vocabulary inferred from `token_embd.weight`'s second dimension.
    pub(super) n_vocab: u32,
    /// Layer count.
    pub(super) n_layer: u32,
    /// Attention head count, using the first array value.
    pub(super) n_head: u32,
    /// KV head count, using the first array value.
    pub(super) n_head_kv: u32,
    /// Expert count, defaulting to zero.
    pub(super) n_expert: u32,
    /// Key head dimension.
    pub(super) n_embd_head_k: u32,
    /// Value head dimension.
    pub(super) n_embd_head_v: u32,
    /// Declared split count, narrowed as in the source helper.
    pub(super) n_split: u16,
    /// Declared total tensor count across splits; mismatches are diagnostic.
    pub(super) n_split_tensors: u32,
    /// Raw first-shard metadata entries, in their original order.
    pub(super) entries: Vec<Entry>,
    /// Tensors in first-shard and then ascending-shard order.
    pub(super) tensors: Vec<Tensor>,
    /// Tensor count in the first shard, before append-only metadata merging.
    pub(super) first_shard_tensors: usize,
}

struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}
impl<'a> Reader<'a> {
    fn take(&mut self, len: usize) -> Option<&'a [u8]> {
        let end = self.pos.checked_add(len)?;
        let bytes = self.bytes.get(self.pos..end)?;
        self.pos = end;
        Some(bytes)
    }
    fn read<const N: usize>(&mut self) -> Option<[u8; N]> {
        self.take(N)?.try_into().ok()
    }
    fn u32(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.read()?))
    }
    fn i32(&mut self) -> Option<i32> {
        Some(i32::from_le_bytes(self.read()?))
    }
    fn u64(&mut self) -> Option<u64> {
        Some(u64::from_le_bytes(self.read()?))
    }
    fn string(&mut self) -> Option<Vec<u8>> {
        let len = usize::try_from(self.u64()?).ok()?;
        Some(self.take(len)?.to_vec())
    }
    fn skip_value(&mut self, kind: i32, depth: usize) -> Option<()> {
        if depth > 64 {
            return None;
        }
        if kind == 8 {
            self.string()?;
            return Some(());
        }
        if kind == 9 {
            let kind = self.i32()?;
            let count = usize::try_from(self.u64()?).ok()?;
            if count > MAX_ENTRIES {
                return None;
            }
            if kind == 8 || kind == 9 {
                for _ in 0..count {
                    self.skip_value(kind, depth + 1)?;
                }
            } else {
                self.take(count.checked_mul(value_size(kind)?)?)?;
            }
        } else {
            self.take(value_size(kind)?)?;
        }
        Some(())
    }
    fn uint(&mut self, kind: i32, depth: usize) -> Option<u32> {
        if depth > 64 {
            return None;
        }
        Some(match kind {
            0 => u32::from(self.read::<1>()?[0]),
            1 => u32::from_le_bytes(i32::from(i8::from_le_bytes(self.read()?)).to_le_bytes()),
            2 => u32::from(u16::from_le_bytes(self.read()?)),
            3 => u32::from_le_bytes(i32::from(i16::from_le_bytes(self.read()?)).to_le_bytes()),
            4 | 5 => self.u32()?,
            10 | 11 => u32::from_le_bytes(self.read::<8>()?[..4].try_into().ok()?),
            9 => {
                let element = self.i32()?;
                let count = usize::try_from(self.u64()?).ok()?;
                if count == 0 || count > MAX_ENTRIES {
                    return None;
                }
                let value = self.uint(element, depth + 1)?;
                for _ in 1..count {
                    self.take(value_size(element)?)?;
                }
                value
            }
            _ => return None,
        })
    }
}
const fn value_size(kind: i32) -> Option<usize> {
    match kind {
        0 | 1 | 7 => Some(1),
        2 | 3 => Some(2),
        4..=6 => Some(4),
        10..=12 => Some(8),
        _ => None,
    }
}

/// Parses exactly the header/KV/tensor domain of the original test helper.
pub(super) fn parse_meta(bytes: &[u8]) -> Option<Model> {
    if bytes.len() > MAX_HEADER {
        return None;
    }
    let mut r = Reader { bytes, pos: 0 };
    if r.take(4)? != b"GGUF" {
        return None;
    }
    let version = r.u32()?;
    if !(2..=3).contains(&version) {
        return None;
    }
    let tensors = usize::try_from(r.u64()?).ok()?;
    let entries = usize::try_from(r.u64()?).ok()?;
    if tensors > MAX_ENTRIES || entries > MAX_ENTRIES {
        return None;
    }
    let mut model = Model {
        version,
        ..Model::default()
    };
    let mut prefix = Vec::new();
    for _ in 0..entries {
        let key = r.string()?;
        let kind = r.i32()?;
        let start = r.pos;
        if key == b"general.architecture" && kind == 8 {
            model.architecture = r.string()?;
            prefix.clone_from(&model.architecture);
            prefix.push(b'.');
        } else if key == b"split.count" {
            model.n_split =
                u16::from_le_bytes(r.uint(kind, 0)?.to_le_bytes()[..2].try_into().ok()?);
        } else if key == b"split.tensors.count" {
            model.n_split_tensors = r.uint(kind, 0)?;
        } else {
            let suffix = (!prefix.is_empty())
                .then(|| key.strip_prefix(prefix.as_slice()))
                .flatten();
            let target = match suffix {
                Some(b"embedding_length") => Some(&mut model.n_embd),
                Some(b"feed_forward_length") => Some(&mut model.n_ff),
                Some(b"block_count") => Some(&mut model.n_layer),
                Some(b"attention.head_count") => Some(&mut model.n_head),
                Some(b"attention.head_count_kv") => Some(&mut model.n_head_kv),
                Some(b"expert_count") => Some(&mut model.n_expert),
                Some(b"attention.key_length") => Some(&mut model.n_embd_head_k),
                Some(b"attention.value_length") => Some(&mut model.n_embd_head_v),
                _ => None,
            };
            if let Some(target) = target {
                *target = r.uint(kind, 0)?;
            } else {
                r.skip_value(kind, 0)?;
            }
        }
        model.entries.push(Entry {
            key,
            kind,
            payload: bytes.get(start..r.pos)?.to_vec(),
        });
    }
    for _ in 0..tensors {
        let name = r.string()?;
        let n_dims = r.u32()?;
        if n_dims > 4 {
            return None;
        }
        let mut dimensions = [1; 4];
        for dimension in dimensions.iter_mut().take(usize::try_from(n_dims).ok()?) {
            *dimension = i64::from_le_bytes(r.read()?);
        }
        let kind = r.i32()?;
        let offset = r.u64()?;
        if name == b"token_embd.weight" {
            model.n_vocab = u32::from_le_bytes(dimensions[1].to_le_bytes()[..4].try_into().ok()?);
        }
        model.tensors.push(Tensor {
            name,
            kind,
            dimensions,
            n_dims,
            offset,
        });
    }
    model.first_shard_tensors = model.tensors.len();
    Some(model)
}

/// Raw HTTP-like response supplied by a verified immutable fixture provider.
pub(super) struct Response {
    /// Original HTTP status or equivalent frozen response status.
    pub(super) status: u16,
    /// Unmodified byte prefix, never generated from expected schemas.
    pub(super) bytes: Vec<u8>,
}
/// Retrieval interface separating the rewritten helper from external assets.
pub(super) trait Provider {
    /// Returns the raw HF sibling-listing response for the requested repository.
    fn listing(&mut self, repo: &str) -> Result<Response>;
    /// Returns an actual GGUF prefix for the requested inclusive-byte range size.
    fn prefix(&mut self, repo: &str, filename: &str, bytes: usize) -> Result<Response>;
}

/// Preserves the original helper's case-sensitive suffix and preference rules.
#[allow(
    clippy::case_sensitive_file_extension_comparisons,
    reason = "The original helper uses the exact lowercase .gguf suffix."
)]
pub(super) fn detect_filename(listing: &[u8], quant: &str) -> Result<Option<(String, String)>> {
    let value: Value = match serde_json::from_slice(listing) {
        Ok(value) => value,
        Err(_) => return Ok(None),
    };
    let Some(siblings) = value.get("siblings").and_then(Value::as_array) else {
        return Ok(None);
    };
    let mut matches = Vec::new();
    let quant = quant.to_ascii_uppercase();
    for sibling in siblings {
        let Some(name) = sibling.get("rfilename") else {
            continue;
        };
        let name = name
            .as_str()
            .ok_or_else(|| invalid("HF sibling filename must be text"))?;
        if name.ends_with(".gguf") && name.to_ascii_uppercase().contains(&quant) {
            matches.push(name.to_owned());
        }
    }
    matches.sort();
    for name in &matches {
        if !name.contains("-of-") && !name.contains("mmproj") {
            return Ok(Some((name.clone(), String::new())));
        }
    }
    for name in &matches {
        if let Some(pos) = name.find("-00001-of-") {
            return Ok(Some((name.clone(), name[..pos].to_owned())));
        }
    }
    Ok(matches.into_iter().next().map(|name| (name, String::new())))
}
/// Matches the source's cache-component substitutions.
pub(super) fn sanitize(value: &str) -> String {
    value
        .chars()
        .map(|c| {
            if matches!(c, '/' | '\\' | ':') {
                '_'
            } else {
                c
            }
        })
        .collect()
}
/// Constructs a source-compatible cache filename beneath an explicit root.
pub(super) fn cache_path(directory: &Path, repo: &str, filename: &str) -> PathBuf {
    directory.join(format!(
        "{}--{}.partial",
        sanitize(repo),
        sanitize(filename)
    ))
}

fn fetch_cached(
    provider: &mut impl Provider,
    directory: &Path,
    repo: &str,
    filename: &str,
) -> Result<Option<Model>> {
    let path = cache_path(directory, repo, filename);
    if let Ok(bytes) = fs::read(&path)
        && !bytes.is_empty()
        && let Some(model) = parse_meta(&bytes)
    {
        return Ok(Some(model));
    }
    let _ = fs::create_dir_all(directory);
    let mut size = 2 * 1024 * 1024;
    while size <= MAX_HEADER {
        let response = provider.prefix(repo, filename, size)?;
        if !matches!(response.status, 200 | 206) || response.bytes.is_empty() {
            return Ok(None);
        }
        if let Some(model) = parse_meta(&response.bytes) {
            let _ = fs::write(&path, &response.bytes);
            return Ok(Some(model));
        }
        if response.status == 200 {
            return Ok(None);
        }
        size *= 2;
    }
    Ok(None)
}

/// Fetches/caches every split and appends tensors in the original shard order.
pub(super) fn fetch_model(
    provider: &mut impl Provider,
    directory: &Path,
    repo: &str,
    quant: &str,
) -> Result<Option<Model>> {
    let listing = provider.listing(repo)?;
    if listing.status != 200 || listing.bytes.is_empty() {
        return Ok(None);
    }
    let Some((filename, prefix)) = detect_filename(&listing.bytes, quant)? else {
        return Ok(None);
    };
    let Some(mut model) = fetch_cached(provider, directory, repo, &filename)? else {
        return Ok(None);
    };
    if model.n_split > 1 {
        if prefix.is_empty() {
            return Ok(None);
        }
        for index in 2..=model.n_split {
            let filename = format!("{prefix}-{index:05}-of-{:05}.gguf", model.n_split);
            let Some(shard) = fetch_cached(provider, directory, repo, &filename)? else {
                return Ok(None);
            };
            model.tensors.extend(shard.tensors);
            if model.tensors.len() > MAX_ENTRIES {
                return Err(invalid("merged GGUF tensor count exceeds bound"));
            }
        }
    }
    // A split.tensors.count mismatch is a warning in the source, not rejection.
    Ok(Some(model))
}

/// Equivalent metadata form of the helper's merged GGUF-context path.
#[allow(
    dead_code,
    reason = "This helper API is also consumed by the graph-export source port; these two source executables use fetch_model instead."
)]
pub(super) fn fetch_context(
    provider: &mut impl Provider,
    directory: &Path,
    repo: &str,
    quant: &str,
) -> Result<Option<Model>> {
    let Some(mut model) = fetch_model(provider, directory, repo, quant)? else {
        return Ok(None);
    };
    if model.n_split > 1 {
        let alignment = model
            .entries
            .iter()
            .find(|entry| entry.key == b"general.alignment")
            .map_or(Some(32), |entry| {
                if entry.kind == 4 {
                    Some(u32::from_le_bytes(
                        entry.payload.as_slice().try_into().ok()?,
                    ))
                } else {
                    None
                }
            });
        let Some(alignment) = alignment.filter(|n| n.is_power_of_two()) else {
            return Ok(None);
        };
        // gguf_add_tensor appends aligned descriptors after the first shard's
        // original descriptors. The replacement owns this metadata in Rust.
        for index in model.first_shard_tensors..model.tensors.len() {
            if index == 0 {
                model.tensors[index].offset = 0;
                continue;
            }
            let previous = &model.tensors[index - 1];
            let Some(bytes) = tensor_bytes(previous) else {
                return Ok(None);
            };
            let alignment = u64::from(alignment);
            let Some(padded) = bytes
                .checked_add(alignment - 1)
                .map(|n| n & !(alignment - 1))
            else {
                return Ok(None);
            };
            let Some(offset) = previous.offset.checked_add(padded) else {
                return Ok(None);
            };
            let tensor = &mut model.tensors[index];
            tensor.offset = offset;
            // Additional tensors pass through ggml_tensor's fixed name buffer.
            tensor.name.truncate(63);
        }
        let mut names = std::collections::BTreeSet::new();
        if model
            .tensors
            .iter()
            .any(|tensor| !names.insert(&tensor.name))
        {
            return Ok(None);
        }
        if let Some(entry) = model
            .entries
            .iter_mut()
            .find(|entry| entry.key == b"split.count")
        {
            entry.kind = 2;
            entry.payload = 1_u16.to_le_bytes().to_vec();
        } else {
            model.entries.push(Entry {
                key: b"split.count".to_vec(),
                kind: 2,
                payload: 1_u16.to_le_bytes().to_vec(),
            });
        }
        model.n_split = 1;
    }
    Ok(Some(model))
}

// GGUF's on-disk block layouts, independent of engine quantization algorithms.
fn tensor_bytes(tensor: &Tensor) -> Option<u64> {
    let (block, bytes): (u64, u64) = match tensor.kind {
        0 | 26 => (1, 4),
        1 | 25 | 30 => (1, 2),
        2 | 20 => (32, 18),
        3 => (32, 20),
        6 => (32, 22),
        7 => (32, 24),
        8 => (32, 34),
        9 => (32, 36),
        10 => (256, 84),
        11 | 21 => (256, 110),
        12 => (256, 144),
        13 => (256, 176),
        14 => (256, 210),
        15 => (256, 292),
        16 | 35 => (256, 66),
        17 => (256, 74),
        18 => (256, 98),
        19 => (256, 50),
        22 => (256, 82),
        23 => (256, 136),
        24 => (1, 1),
        27 | 28 => (1, 8),
        29 => (256, 56),
        34 => (256, 54),
        39 => (32, 17),
        40 => (64, 36),
        41 => (128, 18),
        42 => (64, 18),
        142 => (128, 34),
        143 => (128, 28),
        _ => return None,
    };
    let first = u64::try_from(tensor.dimensions[0]).ok()?;
    if first == 0 || first % block != 0 {
        return None;
    }
    tensor.dimensions[1..]
        .iter()
        .try_fold(first / block * bytes, |n, &dim| {
            let dim = u64::try_from(dim).ok()?;
            (dim > 0).then_some(())?;
            n.checked_mul(dim)
        })
}

/// Verified immutable response provider shared by the two complete source ports.
pub(super) struct Frozen<'a> {
    engine: &'a Engine,
    models: Vec<Value>,
    /// Number of raw prefix retrievals, available as a cache diagnostic.
    pub(super) prefix_reads: usize,
}
impl<'a> Frozen<'a> {
    /// Opens the root-owned immutable HF response manifest.
    pub(super) fn open(engine: &'a Engine) -> Result<Self> {
        let path = engine.asset_path("metadata/manifest.json")?;
        if fs::metadata(&path)?.len() > 1024 * 1024 {
            return Err(invalid("metadata manifest exceeds1MiB"));
        }
        let manifest: Value = serde_json::from_slice(&fs::read(path)?)?;
        exact_fields(&manifest, &["SchemaVersion", "Purpose", "Models"])?;
        field(&manifest, "Purpose")?
            .as_str()
            .filter(|purpose| purpose.len() <= 4096)
            .ok_or_else(|| invalid("metadata corpus purpose must be bounded text"))?;
        if uint(&manifest, "SchemaVersion")? != 1 {
            return Err(invalid("metadata manifest version unsupported"));
        }
        let models = array(&manifest, "Models")?;
        if models.len() != 11 {
            return Err(invalid(
                "all eleven source model metadata inputs are required",
            ));
        }
        for model in models {
            if model.get("ProviderRepo").is_some() || model.get("SourceCorrection").is_some() {
                exact_fields(
                    model,
                    &[
                        "Repo",
                        "Quant",
                        "Revision",
                        "Api",
                        "Files",
                        "ProviderRepo",
                        "SourceCorrection",
                    ],
                )?;
                for name in ["ProviderRepo", "SourceCorrection"] {
                    field(model, name)?
                        .as_str()
                        .ok_or_else(|| invalid("metadata correction must be text"))?;
                }
            } else {
                exact_fields(model, &["Repo", "Quant", "Revision", "Api", "Files"])?;
            }
            for name in ["Repo", "Quant", "Revision"] {
                field(model, name)?
                    .as_str()
                    .ok_or_else(|| invalid("metadata manifest identity must be text"))?;
            }
            let files = array(model, "Files")?;
            if files.is_empty() || files.len() > 256 {
                return Err(invalid("metadata shard manifest extent invalid"));
            }
        }
        Ok(Self {
            engine,
            models: models.to_vec(),
            prefix_reads: 0,
        })
    }
    /// Reports explicit publisher-identity corrections retained in the input lock.
    pub(super) fn corrections(&self) -> Vec<Value> {
        self.models.iter().filter(|model|model.get("SourceCorrection").is_some()).map(|model|serde_json::json!({"Repo":model["Repo"],"ProviderRepo":model["ProviderRepo"],"SourceCorrection":model["SourceCorrection"],"Revision":model["Revision"]})).collect()
    }
    fn model(&self, repo: &str, quant: Option<&str>) -> Result<&Value> {
        self.models
            .iter()
            .find(|model| {
                model["Repo"] == repo && quant.is_none_or(|quant| model["Quant"] == quant)
            })
            .ok_or_else(|| invalid(format!("verified metadata identity missing: {repo}")))
    }
    fn read(&self, record: &Value, file: bool) -> Result<Vec<u8>> {
        if file {
            exact_fields(
                record,
                &[
                    "Filename",
                    "Path",
                    "Url",
                    "Bytes",
                    "Sha256",
                    "Status",
                    "RangeBytes",
                ],
            )?;
        } else if record.get("CorpusPath").is_some() {
            exact_fields(record, &["Path", "Url", "Bytes", "Sha256", "CorpusPath"])?;
        } else {
            exact_fields(record, &["Path", "Url", "Bytes", "Sha256"])?;
        }
        let path = field(record, "Path")?
            .as_str()
            .ok_or_else(|| invalid("metadata artifact path must be text"))?;
        let count = uint(record, "Bytes")?;
        let digest = field(record, "Sha256")?
            .as_str()
            .ok_or_else(|| invalid("metadata artifact digest must be text"))?;
        if count == 0
            || count > u64::try_from(MAX_HEADER)?
            || digest.len() != 64
            || !digest.bytes().all(|byte| byte.is_ascii_hexdigit())
        {
            return Err(invalid("metadata artifact identity outside bounds"));
        }
        let path = self.engine.asset_path(path)?;
        if !path.is_file() || fs::metadata(&path)?.len() != count {
            return Err(invalid("metadata artifact file/length mismatch"));
        }
        let bytes = fs::read(path)?;
        if u64::try_from(bytes.len())? != count {
            return Err(invalid("metadata artifact length mismatch"));
        }
        let hash = Sha256::digest(&bytes);
        let actual = hash
            .iter()
            .flat_map(|byte| [byte >> 4, byte & 15])
            .map(|nibble| char::from(b"0123456789abcdef"[usize::from(nibble)]))
            .collect::<String>();
        if actual != digest.to_ascii_lowercase() {
            return Err(invalid("metadata artifact SHA256 mismatch"));
        }
        Ok(bytes)
    }
    /// Fetches the source repo/quant with exact required identity validation.
    pub(super) fn fetch(
        &mut self,
        directory: &Path,
        repo: &str,
        quant: &str,
    ) -> Result<Option<Model>> {
        self.model(repo, Some(quant))?;
        fetch_model(self, directory, repo, quant)
    }
}
impl Provider for Frozen<'_> {
    fn listing(&mut self, repo: &str) -> Result<Response> {
        let record = field(self.model(repo, None)?, "Api")?;
        Ok(Response {
            status: 200,
            bytes: self.read(record, false)?,
        })
    }
    fn prefix(&mut self, repo: &str, filename: &str, count: usize) -> Result<Response> {
        let record = array(self.model(repo, None)?, "Files")?
            .iter()
            .find(|file| file["Filename"] == filename)
            .ok_or_else(|| invalid(format!("verified GGUF shard absent: {repo}/{filename}")))?;
        let status = u16::try_from(uint(record, "Status")?)?;
        let range = usize::try_from(uint(record, "RangeBytes")?)?;
        if !matches!(status, 200 | 206) || count > range || range > MAX_HEADER {
            return Err(invalid(
                "frozen GGUF range/status does not cover requested bytes",
            ));
        }
        let mut bytes = self.read(record, true)?;
        if status == 206 {
            bytes.truncate(count);
        }
        self.prefix_reads += 1;
        Ok(Response { status, bytes })
    }
}

#[cfg(test)]
mod tests {
    use super::{Result, array, detect_filename, field, invalid, parse_meta};

    #[test]
    fn frozen_listings_select_shards_present_in_the_immutable_lock() -> Result<()> {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/upstream_prism");
        let manifest: serde_json::Value =
            serde_json::from_slice(include_bytes!("metadata-assets.lock.json"))?;
        let models = array(&manifest, "Models")?;
        assert_eq!(models.len(), 11);
        for model in models {
            let text = |record, key| -> Result<&str> {
                field(record, key)?
                    .as_str()
                    .ok_or_else(|| invalid("frozen model identity must be text"))
            };
            let repo = text(model, "Repo")?;
            let quant = text(model, "Quant")?;
            let api = field(model, "Api")?;
            let corpus = text(api, "CorpusPath")?
                .strip_prefix("tools/prism/upstream/")
                .ok_or_else(|| invalid("frozen API corpus path has an unexpected prefix"))?;
            let listing = std::fs::read(root.join(corpus))?;
            let (filename, _) = detect_filename(&listing, quant)?
                .ok_or_else(|| invalid(format!("frozen listing selects no GGUF: {repo}")))?;
            assert!(
                array(model, "Files")?
                    .iter()
                    .any(|file| file["Filename"] == filename),
                "source-selected GGUF missing from immutable lock: {repo}/{filename}"
            );
            if repo == "ggml-org/Qwen3.6-27B-GGUF" {
                // std::string/std::sort orders uppercase Qwen before lowercase
                // dflash. Both real files occur in this unmodified API corpus.
                assert_eq!(filename, "Qwen3.6-27B-Q8_0.gguf");
            }
        }
        Ok(())
    }

    #[test]
    fn truncated_raw_descriptor_never_becomes_a_model() {
        // One independently written valid v3 descriptor, without tensor data.
        let mut header = b"GGUF\x03\0\0\0\x01\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0".to_vec();
        header.extend(1_u64.to_le_bytes());
        header.push(b'x');
        header.extend(2_u32.to_le_bytes());
        header.extend(32_i64.to_le_bytes());
        header.extend(1_i64.to_le_bytes());
        header.extend(8_i32.to_le_bytes());
        header.extend(0_u64.to_le_bytes());
        assert!(parse_meta(&header).is_some());
        for end in 0..header.len() {
            assert!(parse_meta(&header[..end]).is_none());
        }
    }
}
