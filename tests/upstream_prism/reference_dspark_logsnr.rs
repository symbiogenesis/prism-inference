//! Clean-room independent safe Rust `GGUF` v2/v3 reader and `DSpark` log-SNR metadata guard.
//!
//! No production `bonsai_prism_runtime` or `bonsai_gguf` code, native bridge, or
//! precomputed answer fixture is called here. Parses `GGUF` headers, metadata
//! entries, alignment, and tensor descriptors directly using bounded `std::io::Read`.

use std::collections::BTreeSet;
use std::fs::{self, File};
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

use serde_json::{Value, json};

use super::{Result, invalid};

pub(super) const ID: &str = "tests/test-dspark-logsnr-meta.cpp";
const SOURCE_SHA: &str = "2fef2521dbc0985a3aed6bbe50de5a54b7a75dc6c3c5933e04fc8ceb4c7e3112";

const MAX_OPERATIONS: usize = 128;
const MAX_PATHS_PER_OP: usize = 16;
const MAX_PATH_BYTES: usize = 32_768;
const MAX_MODEL_FILE_BYTES: u64 = 64 * 1024 * 1024;
const MAX_MODEL_LOG_BYTES: usize = 128 * 1024;
const MAX_TOTAL_LOG_BYTES: usize = 1024 * 1024;

const DEFAULT_ALIGNMENT: u64 = 32;
const MAX_ALIGNMENT: u32 = 1_048_576;
const MAX_KV_COUNT: u64 = 1_048_576;
const MAX_TENSOR_COUNT: u64 = 1_048_576;
const MAX_STRING_BYTES: u64 = 8 * 1024 * 1024;
const MAX_ARRAY_ITEMS: u64 = 1_048_576;
const MIN_KV_BYTES: u64 = 14;
const MIN_TENSOR_BYTES: u64 = 33;

#[derive(Clone, Debug)]
enum KvValue {
    Bool(bool),
    U32(u32),
    F32(f32),
    String(String),
    Other,
}

struct ObservedMetadata {
    alignment: u64,
    architecture: Option<KvValue>,
    dspark_conditioning: Option<KvValue>,
    dspark_min_log_snr: Option<KvValue>,
    dspark_max_log_snr: Option<KvValue>,
}

impl ObservedMetadata {
    const fn new() -> Self {
        Self {
            alignment: DEFAULT_ALIGNMENT,
            architecture: None,
            dspark_conditioning: None,
            dspark_min_log_snr: None,
            dspark_max_log_snr: None,
        }
    }
}

struct TensorSpan {
    offset: u64,
    bytes: u64,
}

struct ModelOutcome {
    loaded: bool,
    log: String,
}

impl ModelOutcome {
    fn accepted(message: &str) -> Self {
        Self {
            loaded: true,
            log: message.to_owned(),
        }
    }

    fn rejected(message: &str) -> Self {
        Self {
            loaded: false,
            log: message.to_owned(),
        }
    }
}

struct BoundedReader<R> {
    inner: std::io::Take<BufReader<R>>,
    pos: u64,
    len: u64,
}

impl<R: Read> BoundedReader<R> {
    fn new(reader: R, len: u64) -> Self {
        Self {
            inner: BufReader::new(reader).take(len),
            pos: 0,
            len,
        }
    }

    const fn remaining(&self) -> u64 {
        self.len.saturating_sub(self.pos)
    }

    fn read_array<const N: usize>(&mut self) -> std::result::Result<[u8; N], &'static str> {
        let needed = u64::try_from(N).map_err(|_| "gguf_load: read size overflow")?;
        if self.remaining() < needed {
            return Err("gguf_load: unexpected end of file");
        }
        let mut buf = [0_u8; N];
        self.inner
            .read_exact(&mut buf)
            .map_err(|_| "gguf_load: truncated GGUF read")?;
        self.pos += needed;
        Ok(buf)
    }

    fn read_u8(&mut self) -> std::result::Result<u8, &'static str> {
        Ok(self.read_array::<1>()?[0])
    }

    fn read_u32(&mut self) -> std::result::Result<u32, &'static str> {
        Ok(u32::from_le_bytes(self.read_array::<4>()?))
    }

    fn read_u64(&mut self) -> std::result::Result<u64, &'static str> {
        Ok(u64::from_le_bytes(self.read_array::<8>()?))
    }

    fn read_f32(&mut self) -> std::result::Result<f32, &'static str> {
        Ok(f32::from_bits(self.read_u32()?))
    }

    fn read_bytes(
        &mut self,
        count: u64,
        max_bytes: u64,
    ) -> std::result::Result<Vec<u8>, &'static str> {
        if count > max_bytes || count > self.remaining() {
            return Err("gguf_load: byte extent exceeds bound or file length");
        }
        let n = usize::try_from(count).map_err(|_| "gguf_load: byte extent overflow")?;
        let mut buf = vec![0_u8; n];
        self.inner
            .read_exact(&mut buf)
            .map_err(|_| "gguf_load: truncated string or byte payload")?;
        self.pos += count;
        Ok(buf)
    }

    fn skip_bytes(&mut self, count: u64) -> std::result::Result<(), &'static str> {
        if count > self.remaining() {
            return Err("gguf_load: array extent exceeds file length");
        }
        let mut scratch = [0_u8; 4096];
        let mut left = count;
        while left > 0 {
            let chunk =
                usize::try_from(left.min(4096)).map_err(|_| "gguf_load: skip chunk overflow")?;
            self.inner
                .read_exact(&mut scratch[..chunk])
                .map_err(|_| "gguf_load: truncated scalar or array payload")?;
            let step = u64::try_from(chunk).map_err(|_| "gguf_load: skip step overflow")?;
            self.pos += step;
            left -= step;
        }
        Ok(())
    }

    fn read_string(&mut self) -> std::result::Result<String, &'static str> {
        let len = self.read_u64()?;
        let bytes = self.read_bytes(len, MAX_STRING_BYTES)?;
        String::from_utf8(bytes).map_err(|_| "gguf_load: invalid UTF-8 string")
    }
}

const fn scalar_width(kind: u32) -> Option<u64> {
    match kind {
        0 | 1 | 7 => Some(1),
        2 | 3 => Some(2),
        4..=6 => Some(4),
        10..=12 => Some(8),
        _ => None,
    }
}

const fn tensor_layout(kind: u32) -> Option<(u64, u64)> {
    Some(match kind {
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
        _ => return None,
    })
}

fn align_up(value: u64, alignment: u64) -> std::result::Result<u64, &'static str> {
    value
        .checked_add(alignment - 1)
        .map(|aligned| aligned & !(alignment - 1))
        .ok_or("gguf_load: alignment arithmetic overflow")
}

fn parse_kv_value<R: Read>(
    reader: &mut BoundedReader<R>,
    kind: u32,
) -> std::result::Result<KvValue, &'static str> {
    match kind {
        4 => Ok(KvValue::U32(reader.read_u32()?)),
        6 => Ok(KvValue::F32(reader.read_f32()?)),
        7 => Ok(KvValue::Bool(reader.read_u8()? != 0)),
        8 => Ok(KvValue::String(reader.read_string()?)),
        9 => {
            let elem_type = reader.read_u32()?;
            let count = reader.read_u64()?;
            if elem_type == 9 || elem_type > 12 {
                return Err("gguf_load: invalid GGUF array element type");
            }
            if count > MAX_ARRAY_ITEMS {
                return Err("gguf_load: GGUF array item count exceeds bound");
            }
            if elem_type == 8 {
                let min_bytes = count
                    .checked_mul(8)
                    .ok_or("gguf_load: string array extent overflow")?;
                if min_bytes > reader.remaining() {
                    return Err("gguf_load: truncated string array");
                }
                for _ in 0..count {
                    let _ = reader.read_string()?;
                }
            } else {
                let width =
                    scalar_width(elem_type).ok_or("gguf_load: invalid scalar array type")?;
                let total_bytes = count
                    .checked_mul(width)
                    .ok_or("gguf_load: scalar array byte count overflow")?;
                reader.skip_bytes(total_bytes)?;
            }
            Ok(KvValue::Other)
        }
        _ => {
            let width = scalar_width(kind).ok_or("gguf_load: unsupported metadata value type")?;
            reader.skip_bytes(width)?;
            Ok(KvValue::Other)
        }
    }
}

fn parse_kv_entries<R: Read>(
    reader: &mut BoundedReader<R>,
    n_kv: u64,
) -> std::result::Result<ObservedMetadata, &'static str> {
    if n_kv > MAX_KV_COUNT || n_kv > reader.remaining() / MIN_KV_BYTES {
        return Err("gguf_load: metadata entry count exceeds file bounds");
    }
    let mut seen_keys = BTreeSet::new();
    let mut observed = ObservedMetadata::new();
    for _ in 0..n_kv {
        let key = reader.read_string()?;
        if key.is_empty() || key.contains('\0') {
            return Err("gguf_load: empty or NUL metadata key");
        }
        if !seen_keys.insert(key.clone()) {
            return Err("gguf_load: duplicate metadata key");
        }
        let kind = reader.read_u32()?;
        let value = parse_kv_value(reader, kind)?;
        match key.as_str() {
            "general.alignment" => {
                let KvValue::U32(align) = value else {
                    return Err("gguf_load: general.alignment must be UINT32");
                };
                if !align.is_power_of_two() || align > MAX_ALIGNMENT {
                    return Err("gguf_load: invalid general.alignment value");
                }
                observed.alignment = u64::from(align);
            }
            "general.architecture" => observed.architecture = Some(value),
            "dspark.dspark.log_snr_conditioning" => observed.dspark_conditioning = Some(value),
            "dspark.dspark.min_log_snr" => observed.dspark_min_log_snr = Some(value),
            "dspark.dspark.max_log_snr" => observed.dspark_max_log_snr = Some(value),
            _ => {}
        }
    }
    Ok(observed)
}

fn parse_tensor_descriptors<R: Read>(
    reader: &mut BoundedReader<R>,
    n_tensors: u64,
    alignment: u64,
) -> std::result::Result<Vec<TensorSpan>, &'static str> {
    if n_tensors > MAX_TENSOR_COUNT || n_tensors > reader.remaining() / MIN_TENSOR_BYTES {
        return Err("gguf_load: tensor descriptor count exceeds file bounds");
    }
    let count = usize::try_from(n_tensors).map_err(|_| "gguf_load: tensor count overflow")?;
    let mut tensors = Vec::with_capacity(count);
    let mut seen_names = BTreeSet::new();
    for _ in 0..count {
        let name = reader.read_string()?;
        if name.is_empty() || name.len() >= 64 || name.contains('\0') {
            return Err("gguf_load: invalid tensor descriptor name");
        }
        if !seen_names.insert(name) {
            return Err("gguf_load: duplicate tensor descriptor name");
        }
        let rank = reader.read_u32()?;
        if !(1..=4).contains(&rank) {
            return Err("gguf_load: tensor rank outside 1..=4");
        }
        let rank_usize = usize::try_from(rank).map_err(|_| "gguf_load: tensor rank overflow")?;
        let mut dims = [1_u64; 4];
        for dim in dims.iter_mut().take(rank_usize) {
            let extent = reader.read_u64()?;
            if extent == 0 || i64::try_from(extent).is_err() {
                return Err("gguf_load: invalid tensor dimension");
            }
            *dim = extent;
        }
        let kind = reader.read_u32()?;
        let (block, width) =
            tensor_layout(kind).ok_or("gguf_load: unsupported tensor element type")?;
        if !dims[0].is_multiple_of(block) {
            return Err("gguf_load: tensor row width misaligned to block size");
        }
        let elements = dims
            .iter()
            .try_fold(1_u64, |acc, &d| acc.checked_mul(d))
            .filter(|&total| i64::try_from(total).is_ok())
            .ok_or("gguf_load: tensor element count overflow")?;
        let bytes = (elements / block)
            .checked_mul(width)
            .ok_or("gguf_load: tensor byte size overflow")?;
        let offset = reader.read_u64()?;
        if !offset.is_multiple_of(alignment) {
            return Err("gguf_load: unaligned tensor data offset");
        }
        tensors.push(TensorSpan { offset, bytes });
    }
    Ok(tensors)
}

fn validate_tensor_data_bounds(
    descriptors_end: u64,
    file_len: u64,
    alignment: u64,
    mut tensors: Vec<TensorSpan>,
) -> std::result::Result<(), &'static str> {
    let data_offset = if tensors.is_empty() {
        descriptors_end
    } else {
        align_up(descriptors_end, alignment)?
    };
    if data_offset > file_len {
        return Err("gguf_load: tensor data section starts past end of file");
    }
    let available = file_len - data_offset;
    tensors.sort_unstable_by_key(|span| span.offset);
    let mut prev_end = 0_u64;
    for span in tensors {
        if span.offset < prev_end {
            return Err("gguf_load: overlapping tensor data extents");
        }
        let end = span
            .offset
            .checked_add(span.bytes)
            .ok_or("gguf_load: tensor end offset overflow")?;
        if end > available {
            return Err("gguf_load: tensor payload truncated past end of file");
        }
        prev_end = end;
    }
    Ok(())
}

fn validate_dspark_metadata(meta: &ObservedMetadata) -> ModelOutcome {
    let conditioning = match meta.dspark_conditioning {
        Some(KvValue::Bool(flag)) => flag,
        Some(_) => {
            return ModelOutcome::rejected(
                "dspark_load: dspark.dspark.log_snr_conditioning must be boolean",
            );
        }
        None => {
            return ModelOutcome::rejected(
                "dspark_load: missing required dspark.dspark.log_snr_conditioning",
            );
        }
    };
    if !conditioning {
        return ModelOutcome::accepted("dspark_load: accepted unconditioned dspark model metadata");
    }
    let min = match meta.dspark_min_log_snr {
        Some(KvValue::F32(value)) => value,
        Some(_) => {
            return ModelOutcome::rejected("dspark_load: dspark.dspark.min_log_snr must be F32");
        }
        None => {
            return ModelOutcome::rejected(
                "dspark_load: missing required dspark.dspark.min_log_snr for log-SNR conditioning",
            );
        }
    };
    let max = match meta.dspark_max_log_snr {
        Some(KvValue::F32(value)) => value,
        Some(_) => {
            return ModelOutcome::rejected("dspark_load: dspark.dspark.max_log_snr must be F32");
        }
        None => {
            return ModelOutcome::rejected(
                "dspark_load: missing required dspark.dspark.max_log_snr for log-SNR conditioning",
            );
        }
    };
    if !min.is_finite() || !max.is_finite() {
        return ModelOutcome::rejected(
            "dspark_load: non-finite min_log_snr or max_log_snr in dspark metadata",
        );
    }
    if min >= max {
        return ModelOutcome::rejected(
            "dspark_load: invalid log_snr range; require min_log_snr < max_log_snr",
        );
    }
    ModelOutcome::accepted("dspark_load: accepted conditioned dspark model metadata")
}

fn parse_and_validate_gguf<R: Read>(reader: R, file_len: u64) -> ModelOutcome {
    let mut reader = BoundedReader::new(reader, file_len);
    let parsed: std::result::Result<ObservedMetadata, &'static str> = (|| {
        let magic = reader.read_array::<4>()?;
        if &magic != b"GGUF" {
            return Err("gguf_load: invalid GGUF magic header");
        }
        let version = reader.read_u32()?;
        if !matches!(version, 2 | 3) {
            return Err("gguf_load: unsupported GGUF format version");
        }
        let n_tensors = reader.read_u64()?;
        let n_kv = reader.read_u64()?;
        let observed = parse_kv_entries(&mut reader, n_kv)?;
        let tensors = parse_tensor_descriptors(&mut reader, n_tensors, observed.alignment)?;
        validate_tensor_data_bounds(reader.pos, file_len, observed.alignment, tensors)?;
        Ok(observed)
    })();
    let observed = match parsed {
        Ok(meta) => meta,
        Err(message) => return ModelOutcome::rejected(message),
    };
    let Some(KvValue::String(arch)) = &observed.architecture else {
        return ModelOutcome::rejected("gguf_load: missing or non-string general.architecture");
    };
    if arch.is_empty() || arch.contains('\0') {
        return ModelOutcome::rejected("gguf_load: empty or invalid general.architecture");
    }
    if arch == "dspark" {
        return validate_dspark_metadata(&observed);
    }
    ModelOutcome::accepted("gguf_load: accepted model metadata")
}

fn observe_model(path: &Path) -> ModelOutcome {
    let Ok(file) = File::open(path) else {
        return ModelOutcome::rejected("gguf_load: failed to open model file");
    };
    let Ok(meta) = file.metadata() else {
        return ModelOutcome::rejected("gguf_load: failed to stat model file");
    };
    if !meta.is_file() || meta.len() > MAX_MODEL_FILE_BYTES {
        return ModelOutcome::rejected("gguf_load: model file is not regular or exceeds 64 MiB");
    }
    parse_and_validate_gguf(file, meta.len())
}

fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        out.push(char::from(DIGITS[usize::from(byte >> 4)]));
        out.push(char::from(DIGITS[usize::from(byte & 15)]));
    }
    out
}

fn validate_log_bounds(model_bytes: usize, total_bytes: &mut usize) -> Result<()> {
    if model_bytes > MAX_MODEL_LOG_BYTES {
        return Err(invalid("production model log exceeded 128 KiB"));
    }
    let next_total = total_bytes
        .checked_add(model_bytes)
        .filter(|&sum| sum <= MAX_TOTAL_LOG_BYTES)
        .ok_or_else(|| invalid("production model logs exceeded 1 MiB"))?;
    *total_bytes = next_total;
    Ok(())
}

fn exact_fields(value: &Value, expected: &[&str]) -> Result<()> {
    let obj = value
        .as_object()
        .ok_or_else(|| invalid("model observation fields mismatch"))?;
    if obj.len() != expected.len() || expected.iter().any(|key| !obj.contains_key(*key)) {
        return Err(invalid("model observation fields mismatch"));
    }
    Ok(())
}

fn validate_path(value: &Value) -> Result<PathBuf> {
    let text = value
        .as_str()
        .ok_or_else(|| invalid("model observation path must be string"))?;
    if text.is_empty() || text.len() > MAX_PATH_BYTES || text.contains('\0') {
        return Err(invalid("model observation path extent/NUL violation"));
    }
    let path = Path::new(text);
    if !path.is_absolute() {
        return Err(invalid(
            "model observation file must be regular/absolute/bounded64MiB",
        ));
    }
    let meta = fs::metadata(path)
        .map_err(|_| invalid("model observation file must be regular/absolute/bounded64MiB"))?;
    if !meta.is_file() || meta.len() > MAX_MODEL_FILE_BYTES {
        return Err(invalid(
            "model observation file must be regular/absolute/bounded64MiB",
        ));
    }
    Ok(path.to_path_buf())
}

fn validate_operation(operation: &Value) -> Result<Vec<PathBuf>> {
    exact_fields(operation, &["Kind", "Paths", "GpuLayers"])?;
    if operation["Kind"].as_str() != Some("model-load-observe") {
        return Err(invalid("unknown model observation kind"));
    }
    let gpu_layers = operation["GpuLayers"]
        .as_i64()
        .ok_or_else(|| invalid("model observation GPU layers outside 0..=999"))?;
    if !(0..=999).contains(&gpu_layers) {
        return Err(invalid("model observation GPU layers outside 0..=999"));
    }
    let paths = operation["Paths"]
        .as_array()
        .ok_or_else(|| invalid("model observation path count outside 1..=16"))?;
    if paths.is_empty() || paths.len() > MAX_PATHS_PER_OP {
        return Err(invalid("model observation path count outside 1..=16"));
    }
    paths.iter().map(validate_path).collect()
}

pub(super) fn handles(request: &Value) -> bool {
    request["Operations"].as_array().is_some_and(|operations| {
        !operations.is_empty()
            && operations
                .iter()
                .all(|operation| operation["Kind"] == "model-load-observe")
    })
}

pub(super) fn execute(request: &Value) -> Result<Value> {
    exact_fields(request, &["SchemaVersion", "Operations"])?;
    if request["SchemaVersion"].as_i64() != Some(1) {
        return Err(invalid("unsupported SchemaVersion"));
    }
    let operations = request["Operations"]
        .as_array()
        .ok_or_else(|| invalid("Operations must contain 1 through 128 operations"))?;
    if operations.is_empty() || operations.len() > MAX_OPERATIONS {
        return Err(invalid("Operations must contain 1 through 128 operations"));
    }
    let validated_ops = operations
        .iter()
        .map(validate_operation)
        .collect::<Result<Vec<_>>>()?;
    let mut results = Vec::with_capacity(validated_ops.len());
    for paths in validated_ops {
        let mut models = Vec::with_capacity(paths.len());
        let mut total_log_bytes = 0_usize;
        for path in paths {
            let outcome = observe_model(&path);
            validate_log_bounds(outcome.log.len(), &mut total_log_bytes)?;
            models.push(json!({
                "Loaded": outcome.loaded,
                "LogHex": hex(outcome.log.as_bytes()),
            }));
        }
        results.push(json!({"Models": models}));
    }
    Ok(json!({"SchemaVersion": 1, "Results": results}))
}

pub(super) fn identity() -> Value {
    json!({
        "Kind": "independent-rust-dspark-logsnr",
        "SupportedSourceIds": [ID],
        "UpstreamCommit": super::UPSTREAM,
        "AssertionSource": {"Path": ID, "Sha256": SOURCE_SHA},
        "Architecture": "clean-room-gguf-v2-v3-reader-and-dspark-log-snr-metadata-guard-v1",
        "NativeReferenceReplay": false,
        "ProductionRuntimeCalled": false,
        "CompleteSourceGate": false,
        "NativeRetirementQualified": false
    })
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;
    use crate::engine::{Engine, Target};

    struct TempDir(PathBuf);

    impl TempDir {
        fn new(prefix: &str) -> Result<Self> {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
            let seq = NEXT.fetch_add(1, Ordering::Relaxed);
            let dir =
                std::env::temp_dir().join(format!("{prefix}-{}-{nonce}-{seq}", std::process::id()));
            if !dir.is_absolute() {
                return Err(invalid("temp directory must be absolute"));
            }
            fs::create_dir(&dir)?;
            Ok(Self(dir))
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn encode_gguf_string(out: &mut Vec<u8>, s: &str) -> Result<()> {
        out.extend_from_slice(&u64::try_from(s.len())?.to_le_bytes());
        out.extend_from_slice(s.as_bytes());
        Ok(())
    }

    fn push_kv_str(out: &mut Vec<u8>, key: &str, val: &str) -> Result<()> {
        encode_gguf_string(out, key)?;
        out.extend_from_slice(&8_u32.to_le_bytes());
        encode_gguf_string(out, val)?;
        Ok(())
    }

    fn push_kv_u32(out: &mut Vec<u8>, key: &str, val: u32) -> Result<()> {
        encode_gguf_string(out, key)?;
        out.extend_from_slice(&4_u32.to_le_bytes());
        out.extend_from_slice(&val.to_le_bytes());
        Ok(())
    }

    fn push_kv_bool(out: &mut Vec<u8>, key: &str, val: bool) -> Result<()> {
        encode_gguf_string(out, key)?;
        out.extend_from_slice(&7_u32.to_le_bytes());
        out.push(u8::from(val));
        Ok(())
    }

    fn push_kv_f32(out: &mut Vec<u8>, key: &str, val: f32) -> Result<()> {
        encode_gguf_string(out, key)?;
        out.extend_from_slice(&6_u32.to_le_bytes());
        out.extend_from_slice(&val.to_bits().to_le_bytes());
        Ok(())
    }

    fn decode_ascii_hex(hex_str: &str) -> Result<String> {
        if !hex_str.len().is_multiple_of(2)
            || hex_str
                .bytes()
                .any(|b| !b.is_ascii_digit() && !(b'a'..=b'f').contains(&b))
        {
            return Err(invalid("expected lowercase hexadecimal log"));
        }
        let bytes = hex::decode(hex_str)?;
        Ok(String::from_utf8(bytes)?)
    }

    fn assert_no_log_snr(observation: &Value) -> Result<()> {
        assert_eq!(observation["Loaded"], false);
        let text = decode_ascii_hex(
            observation["LogHex"]
                .as_str()
                .ok_or_else(|| invalid("missing LogHex"))?,
        )?;
        assert!(
            !text.contains("log_snr") && !text.contains("log-SNR"),
            "unexpected log_snr mention in structural error: {text}"
        );
        Ok(())
    }

    #[test]
    fn independent_dspark_logsnr_passes_complete_suite_for_reference_and_cpp_targets() -> Result<()>
    {
        let ref_dir = TempDir::new("SwiftBonsai-RefDsparkLogSnr")?;
        let cpp_dir = TempDir::new("SwiftBonsai-CppDsparkLogSnr")?;
        for (target, dir) in [
            (Target::RustReference, ref_dir.path()),
            (Target::Cpp, cpp_dir.path()),
        ] {
            let engine = Engine::new(target, None, None, Some(dir.to_path_buf()), None, None)?;
            let report = crate::dspark_logsnr::run(&engine)?;
            assert_eq!(report["Id"], ID);
            assert_eq!(report["Status"], "pass", "{report}");
        }
        assert!(crate::reference_provider::supports(ID));
        let info = identity();
        assert_eq!(info["SupportedSourceIds"], json!([ID]));
        assert_eq!(info["NativeReferenceReplay"], false);
        assert_eq!(info["ProductionRuntimeCalled"], false);
        assert_eq!(info["CompleteSourceGate"], false);
        assert_eq!(info["NativeRetirementQualified"], false);
        Ok(())
    }

    #[test]
    fn independent_dspark_logsnr_rejects_malformed_protocol_requests() -> Result<()> {
        let dir = TempDir::new("SwiftBonsai-DsparkProto")?;
        let valid_file = dir.path().join("sample.gguf");
        fs::write(&valid_file, b"GGUF")?;
        let valid_str = valid_file
            .to_str()
            .ok_or_else(|| invalid("non-UTF8 temp path"))?
            .to_owned();

        assert!(handles(&json!({
            "SchemaVersion": 1,
            "Operations": [{"Kind": "model-load-observe", "Paths": [valid_str], "GpuLayers": 0}]
        })));
        assert!(!handles(&json!({})));
        assert!(!handles(&json!({"Operations": []})));
        assert!(!handles(&json!({"Operations": [{"Kind": "gguf-read"}]})));
        assert!(!handles(&json!({
            "Operations": [
                {"Kind": "model-load-observe", "Paths": [valid_str], "GpuLayers": 0},
                {"Kind": "gguf-read"}
            ]
        })));

        let base = json!({
            "SchemaVersion": 1,
            "Operations": [{
                "Kind": "model-load-observe",
                "Paths": [valid_str],
                "GpuLayers": 0
            }]
        });

        for layers in [0, 999] {
            let mut req = base.clone();
            req["Operations"][0]["GpuLayers"] = json!(layers);
            let res = execute(&req)?;
            assert_eq!(res["Results"][0]["Models"][0]["Loaded"], false);
        }
        for bad_version in [json!(0), json!(2), json!(1.0), json!(true), json!("1")] {
            let mut req = base.clone();
            req["SchemaVersion"] = bad_version;
            assert!(execute(&req).is_err());
        }
        let mut req_extra_top = base.clone();
        req_extra_top["Extra"] = json!(true);
        assert!(execute(&req_extra_top).is_err());

        for bad_ops in [json!([]), json!(vec![base["Operations"][0].clone(); 129])] {
            let mut req = base.clone();
            req["Operations"] = bad_ops;
            assert!(execute(&req).is_err());
        }
        for bad_layers in [json!(-1), json!(1000), json!(0.0), json!(true), json!("0")] {
            let mut req = base.clone();
            req["Operations"][0]["GpuLayers"] = bad_layers;
            assert!(execute(&req).is_err());
        }

        let mut req_extra_field = base.clone();
        req_extra_field["Operations"][0]["Extra"] = json!(1);
        assert!(execute(&req_extra_field).is_err());

        let mut req_missing_field = base.clone();
        req_missing_field["Operations"][0]
            .as_object_mut()
            .ok_or_else(|| invalid("expected operation object"))?
            .remove("GpuLayers");
        assert!(execute(&req_missing_field).is_err());

        let mut req_bad_kind = base;
        req_bad_kind["Operations"][0]["Kind"] = json!("gguf-read");
        assert!(execute(&req_bad_kind).is_err());
        Ok(())
    }

    #[test]
    fn independent_dspark_logsnr_rejects_invalid_paths_and_oversized_logs() -> Result<()> {
        let dir = TempDir::new("SwiftBonsai-DsparkPaths")?;
        let valid_file = dir.path().join("sample.gguf");
        fs::write(&valid_file, b"GGUF")?;
        let valid_str = valid_file
            .to_str()
            .ok_or_else(|| invalid("non-UTF8 temp path"))?
            .to_owned();
        let base = json!({
            "SchemaVersion": 1,
            "Operations": [{
                "Kind": "model-load-observe",
                "Paths": [valid_str],
                "GpuLayers": 0
            }]
        });

        let mut empty_paths = base.clone();
        empty_paths["Operations"][0]["Paths"] = json!([]);
        assert!(execute(&empty_paths).is_err());

        let mut too_many_paths = base.clone();
        too_many_paths["Operations"][0]["Paths"] = json!(vec![valid_str.clone(); 17]);
        assert!(execute(&too_many_paths).is_err());

        let missing_path = dir
            .path()
            .join("missing.gguf")
            .to_str()
            .ok_or_else(|| invalid("non-UTF8 temp path"))?
            .to_owned();
        let dir_str = dir
            .path()
            .to_str()
            .ok_or_else(|| invalid("non-UTF8 temp path"))?
            .to_owned();
        for bad_path in [
            json!(123),
            json!(""),
            json!("relative/model.gguf"),
            json!(missing_path),
            json!(dir_str),
            json!(format!("{valid_str}\0tail")),
            json!("x".repeat(32_769)),
        ] {
            let mut req = base.clone();
            req["Operations"][0]["Paths"] = json!([bad_path]);
            assert!(execute(&req).is_err());
        }

        let mut agg = 0_usize;
        assert!(validate_log_bounds(MAX_MODEL_LOG_BYTES + 1, &mut agg).is_err());
        for _ in 0..8 {
            validate_log_bounds(MAX_MODEL_LOG_BYTES, &mut agg)?;
        }
        assert!(validate_log_bounds(1, &mut agg).is_err());
        Ok(())
    }

    fn write_temp_gguf(dir: &Path, name: &str, bytes: &[u8]) -> Result<PathBuf> {
        let path = dir.join(name);
        fs::write(&path, bytes)?;
        Ok(path)
    }

    fn gguf_header(version: u32, n_tensors: u64, n_kv: u64) -> Vec<u8> {
        let mut out = b"GGUF".to_vec();
        out.extend_from_slice(&version.to_le_bytes());
        out.extend_from_slice(&n_tensors.to_le_bytes());
        out.extend_from_slice(&n_kv.to_le_bytes());
        out
    }

    #[test]
    fn independent_dspark_logsnr_rejects_corrupt_gguf_without_logsnr_mention() -> Result<()> {
        let dir = TempDir::new("SwiftBonsai-DsparkCorrupt")?;
        let mut no_arch = gguf_header(3, 0, 1);
        push_kv_u32(&mut no_arch, "general.file_type", 0)?;

        let mut bad_arch = gguf_header(3, 0, 1);
        push_kv_u32(&mut bad_arch, "general.architecture", 42)?;

        let mut trunc_tensor = gguf_header(3, 1, 1);
        push_kv_str(&mut trunc_tensor, "general.architecture", "llama")?;
        encode_gguf_string(&mut trunc_tensor, "blk.0.weight")?;
        trunc_tensor.extend_from_slice(&1_u32.to_le_bytes());
        trunc_tensor.extend_from_slice(&4_u64.to_le_bytes());
        trunc_tensor.extend_from_slice(&0_u32.to_le_bytes());
        trunc_tensor.extend_from_slice(&0_u64.to_le_bytes());
        let data_start = usize::try_from(align_up(u64::try_from(trunc_tensor.len())?, 32)?)?;
        trunc_tensor.resize(data_start + 12, 0);

        let mut nested_arr = gguf_header(3, 0, 2);
        push_kv_str(&mut nested_arr, "general.architecture", "llama")?;
        encode_gguf_string(&mut nested_arr, "bad.array")?;
        nested_arr.extend_from_slice(&9_u32.to_le_bytes());
        nested_arr.extend_from_slice(&9_u32.to_le_bytes());
        nested_arr.extend_from_slice(&0_u64.to_le_bytes());

        let mut bad_align = gguf_header(3, 0, 2);
        push_kv_str(&mut bad_align, "general.architecture", "llama")?;
        push_kv_u32(&mut bad_align, "general.alignment", 24)?;

        let paths = vec![
            write_temp_gguf(dir.path(), "empty.gguf", &[])?,
            write_temp_gguf(dir.path(), "bad-magic.gguf", b"NOTGGUF\x03\0\0\0")?,
            write_temp_gguf(dir.path(), "version1.gguf", &gguf_header(1, 0, 0))?,
            write_temp_gguf(dir.path(), "no-arch.gguf", &no_arch)?,
            write_temp_gguf(dir.path(), "bad-arch-type.gguf", &bad_arch)?,
            write_temp_gguf(dir.path(), "truncated-tensor.gguf", &trunc_tensor)?,
            write_temp_gguf(dir.path(), "nested-array.gguf", &nested_arr)?,
            write_temp_gguf(dir.path(), "bad-align.gguf", &bad_align)?,
        ];

        let response = execute(&json!({
            "SchemaVersion": 1,
            "Operations": [{"Kind": "model-load-observe", "Paths": paths, "GpuLayers": 0}]
        }))?;
        let models = response["Results"][0]["Models"]
            .as_array()
            .ok_or_else(|| invalid("missing Models array"))?;
        assert_eq!(models.len(), 8);
        for model in models {
            assert_no_log_snr(model)?;
        }
        Ok(())
    }

    #[test]
    fn independent_dspark_logsnr_handles_v2_kv_types_and_dspark_edge_cases() -> Result<()> {
        let dir = TempDir::new("SwiftBonsai-DsparkKv")?;

        let mut valid_v2 = gguf_header(2, 1, 13);
        push_kv_str(&mut valid_v2, "general.architecture", "llama")?;
        push_kv_u32(&mut valid_v2, "general.alignment", 64)?;
        push_kv_bool(&mut valid_v2, "llama.use_cache", true)?;
        for (key, kind, bytes) in [
            ("test.u8", 0_u32, vec![7_u8]),
            ("test.i8", 1_u32, vec![0xff_u8]),
            ("test.u16", 2_u32, 123_u16.to_le_bytes().to_vec()),
            ("test.i16", 3_u32, (-123_i16).to_le_bytes().to_vec()),
            ("test.i32", 5_u32, (-456_i32).to_le_bytes().to_vec()),
            ("test.u64", 10_u32, 999_u64.to_le_bytes().to_vec()),
            ("test.i64", 11_u32, (-999_i64).to_le_bytes().to_vec()),
            ("test.f64", 12_u32, 3.5_f64.to_bits().to_le_bytes().to_vec()),
        ] {
            encode_gguf_string(&mut valid_v2, key)?;
            valid_v2.extend_from_slice(&kind.to_le_bytes());
            valid_v2.extend_from_slice(&bytes);
        }
        encode_gguf_string(&mut valid_v2, "tokenizer.ggml.tokens")?;
        valid_v2.extend_from_slice(&9_u32.to_le_bytes());
        valid_v2.extend_from_slice(&8_u32.to_le_bytes());
        valid_v2.extend_from_slice(&2_u64.to_le_bytes());
        encode_gguf_string(&mut valid_v2, "a")?;
        encode_gguf_string(&mut valid_v2, "b")?;
        encode_gguf_string(&mut valid_v2, "llama.scores")?;
        valid_v2.extend_from_slice(&9_u32.to_le_bytes());
        valid_v2.extend_from_slice(&6_u32.to_le_bytes());
        valid_v2.extend_from_slice(&1_u64.to_le_bytes());
        valid_v2.extend_from_slice(&1.0_f32.to_bits().to_le_bytes());
        encode_gguf_string(&mut valid_v2, "w")?;
        valid_v2.extend_from_slice(&1_u32.to_le_bytes());
        valid_v2.extend_from_slice(&2_u64.to_le_bytes());
        valid_v2.extend_from_slice(&0_u32.to_le_bytes());
        valid_v2.extend_from_slice(&0_u64.to_le_bytes());
        let v2_start = usize::try_from(align_up(u64::try_from(valid_v2.len())?, 64)?)?;
        valid_v2.resize(v2_start + 8, 0);

        let mut missing_cond = gguf_header(3, 0, 1);
        push_kv_str(&mut missing_cond, "general.architecture", "dspark")?;

        let mut wrong_cond = gguf_header(3, 0, 2);
        push_kv_str(&mut wrong_cond, "general.architecture", "dspark")?;
        push_kv_u32(&mut wrong_cond, "dspark.dspark.log_snr_conditioning", 1)?;

        let mut wrong_min = gguf_header(3, 0, 4);
        push_kv_str(&mut wrong_min, "general.architecture", "dspark")?;
        push_kv_bool(&mut wrong_min, "dspark.dspark.log_snr_conditioning", true)?;
        push_kv_u32(&mut wrong_min, "dspark.dspark.min_log_snr", 1)?;
        push_kv_f32(&mut wrong_min, "dspark.dspark.max_log_snr", 6.0)?;

        let mut wrong_max = gguf_header(3, 0, 4);
        push_kv_str(&mut wrong_max, "general.architecture", "dspark")?;
        push_kv_bool(&mut wrong_max, "dspark.dspark.log_snr_conditioning", true)?;
        push_kv_f32(&mut wrong_max, "dspark.dspark.min_log_snr", -9.0)?;
        push_kv_u32(&mut wrong_max, "dspark.dspark.max_log_snr", 6)?;

        let mut neg_inf_min = gguf_header(3, 0, 4);
        push_kv_str(&mut neg_inf_min, "general.architecture", "dspark")?;
        push_kv_bool(&mut neg_inf_min, "dspark.dspark.log_snr_conditioning", true)?;
        push_kv_f32(
            &mut neg_inf_min,
            "dspark.dspark.min_log_snr",
            f32::NEG_INFINITY,
        )?;
        push_kv_f32(&mut neg_inf_min, "dspark.dspark.max_log_snr", 6.0)?;

        let paths = vec![
            write_temp_gguf(dir.path(), "valid-v2-llama.gguf", &valid_v2)?,
            write_temp_gguf(dir.path(), "dspark-missing-cond.gguf", &missing_cond)?,
            write_temp_gguf(dir.path(), "dspark-wrong-cond-type.gguf", &wrong_cond)?,
            write_temp_gguf(dir.path(), "dspark-wrong-min-type.gguf", &wrong_min)?,
            write_temp_gguf(dir.path(), "dspark-wrong-max-type.gguf", &wrong_max)?,
            write_temp_gguf(dir.path(), "dspark-neg-inf.gguf", &neg_inf_min)?,
        ];

        let response = execute(&json!({
            "SchemaVersion": 1,
            "Operations": [{"Kind": "model-load-observe", "Paths": paths, "GpuLayers": 0}]
        }))?;
        let models = response["Results"][0]["Models"]
            .as_array()
            .ok_or_else(|| invalid("missing check Models array"))?;
        assert_eq!(models[0]["Loaded"], true);
        for model in &models[1..] {
            assert_eq!(model["Loaded"], false);
            let log = decode_ascii_hex(
                model["LogHex"]
                    .as_str()
                    .ok_or_else(|| invalid("missing LogHex"))?,
            )?;
            assert!(log.contains("log_snr") || log.contains("log-SNR"), "{log}");
        }
        Ok(())
    }
}
