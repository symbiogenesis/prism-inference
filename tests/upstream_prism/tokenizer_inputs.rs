//! Verified publisher inputs and Rust-owned vocabulary fixture materialization.
use super::engine::Engine;
use super::{Result, invalid};
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

const LOCK: &str = include_str!("tokenizer-reference-assets.json");
const PREFIX: &str = "tokenizer-reference/";
const MAX_INPUT: u64 = 32 * 1024 * 1024;
const MAX_TOTAL: u64 = 192 * 1024 * 1024;
pub(super) struct Case {
    pub(super) name: String,
    pub(super) vocabulary: PathBuf,
    pub(super) reference_directory: PathBuf,
}
pub(super) struct Prepared {
    pub(super) cases: Vec<Case>,
    pub(super) receipt: Value,
}
fn string<'a>(value: &'a Value, key: &str) -> Result<&'a str> {
    value
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| invalid(format!("Tokenizer input field {key} missing")))
}
fn number(value: &Value, key: &str) -> Result<u64> {
    value
        .get(key)
        .and_then(Value::as_u64)
        .ok_or_else(|| invalid(format!("Tokenizer input field {key} missing")))
}
fn read_locked(path: &Path, length: u64, sha: &str) -> Result<Vec<u8>> {
    if !path.is_absolute()
        || length > MAX_INPUT
        || sha.len() != 64
        || !sha
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err(invalid(
            "Tokenizer locked input path/identity outside bounds",
        ));
    }
    let mut bytes = Vec::new();
    File::open(path)?.take(length + 1).read_to_end(&mut bytes)?;
    if u64::try_from(bytes.len())? != length || super::lora_inputs::hash_bytes(&bytes) != sha {
        return Err(invalid(format!(
            "Tokenizer input identity differs: {}",
            path.display()
        )));
    }
    Ok(bytes)
}
fn write_new(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = OpenOptions::new().create_new(true).write(true).open(path)?;
    file.write_all(bytes)?;
    file.flush()?;
    Ok(())
}
fn filename<'a>(path: &'a str, name: &str) -> Result<&'a str> {
    let Some((prefix, file)) = path.split_once('/') else {
        return Err(invalid("Tokenizer input requires case/filename"));
    };
    if prefix != name
        || !matches!(
            file,
            "config.json"
                | "tokenizer.json"
                | "tokenizer_config.json"
                | "tokenizer_config.source.json"
                | "special_tokens_map.json"
                | "added_tokens.json"
                | "tokenizer.model"
                | "vocab.json"
                | "vocab.txt"
                | "merges.txt"
        )
    {
        return Err(invalid(
            "Tokenizer input path outside original file allowlist",
        ));
    }
    Ok(file)
}
/// Apply a bounded input-only exact reconstruction and verify its final identity.
pub(super) fn materialize(bytes: &[u8], row: &Value) -> Result<Vec<u8>> {
    let Some(transformation) = row.get("Transformation") else {
        return Ok(bytes.to_vec());
    };
    if transformation["Kind"] != "exact-byte-reconstruction" {
        return Err(invalid("Unknown tokenizer input reconstruction"));
    }
    let steps = transformation["Steps"]
        .as_array()
        .filter(|v| !v.is_empty() && v.len() <= 8)
        .ok_or_else(|| invalid("Tokenizer reconstruction steps outside bounds"))?;
    let mut output = std::str::from_utf8(bytes)?.to_owned();
    for step in steps {
        match string(step, "Kind")? {
            "remove-exactly-one-whole-line" => {
                let prefix = string(step, "LinePrefix")?;
                if prefix.is_empty() || prefix.len() > 128 {
                    return Err(invalid("Tokenizer removed-line prefix malformed"));
                }
                let matches = output
                    .split_inclusive('\n')
                    .filter(|line| line.starts_with(prefix))
                    .collect::<Vec<_>>();
                if matches.len() != 1
                    || u64::try_from(matches[0].len())? != number(step, "RemovedBytes")?
                {
                    return Err(invalid("Tokenizer exact removed-line identity differs"));
                }
                output = output
                    .split_inclusive('\n')
                    .filter(|line| !line.starts_with(prefix))
                    .collect();
            }
            "replace-exactly-once" => {
                let from = string(step, "From")?;
                let to = string(step, "To")?;
                if from.is_empty()
                    || from.len() > 128
                    || to.len() > 256
                    || output.matches(from).count() != 1
                {
                    return Err(invalid("Tokenizer exact replacement occurrence differs"));
                }
                output = output.replace(from, to);
            }
            _ => return Err(invalid("Unknown tokenizer input transform step")),
        }
        if u64::try_from(output.len())? > MAX_INPUT {
            return Err(invalid("Tokenizer transformed input exceeds bound"));
        }
    }
    let target = row
        .get("Target")
        .ok_or_else(|| invalid("Tokenizer reconstruction target missing"))?;
    if u64::try_from(output.len())? != number(target, "Bytes")?
        || super::lora_inputs::hash_bytes(output.as_bytes()) != string(target, "Sha256")?
        || target["ExactOriginal"] != true
    {
        return Err(invalid(
            "Tokenizer reconstructed bytes differ from full original identity",
        ));
    }
    Ok(output.into_bytes())
}
#[allow(
    clippy::too_many_lines,
    reason = "Keep complete26-case source and input identity verification together"
)]
pub(super) fn prepare(engine: &Engine) -> Result<Prepared> {
    let manifest: Value = serde_json::from_str(LOCK)?;
    if manifest["SchemaVersion"] != 1
        || manifest["AssetPrefix"] != PREFIX
        || manifest["UpstreamRevision"] != "b87ca7225456b06b0975b8239c66b5aecf07e37c"
    {
        return Err(invalid("Tokenizer input lock provenance mismatch"));
    }
    let cases = manifest["Cases"]
        .as_array()
        .filter(|v| v.len() == 26)
        .ok_or_else(|| invalid("Tokenizer input lock must retain all26 cases"))?;
    let files = manifest["Files"]
        .as_array()
        .filter(|v| v.len() == 122)
        .ok_or_else(|| invalid("Tokenizer input lock file count outside bounds"))?;
    let total = files.iter().try_fold(0_u64, |sum, file| {
        sum.checked_add(number(file, "Bytes")?)
            .filter(|sum| *sum <= MAX_TOTAL)
            .ok_or_else(|| invalid("Tokenizer total input bytes exceed192MiB"))
    })?;
    let mut identities = BTreeMap::new();
    for row in files {
        if identities.insert(string(row, "Path")?, row).is_some() {
            return Err(invalid("Tokenizer input lock duplicate path"));
        }
    }
    let evidence = engine.evidence_path("tokenizer-inputs")?;
    let mut documents = Vec::new();
    for path in [
        "tests/test-tokenizer-0.py",
        "tests/test-tokenizer-0.sh",
        "tests/test-tokenizer-random.py",
    ] {
        let source = engine.source_path(path)?;
        let (length, sha) = super::lora_inputs::digest(&source)?;
        if manifest["Sources"][path] != sha {
            return Err(invalid("Tokenizer original source document hash differs"));
        }
        documents.push(json!({"Path":path,"Bytes":length,"Sha256":sha}));
    }
    let mut result = Vec::new();
    let mut receipts = Vec::new();
    let mut consumed = BTreeSet::new();
    let mut generated = 0;
    for (case, expected) in cases.iter().zip(super::tokenizer_random::NAMES) {
        let name = string(case, "Name")?;
        if name != expected {
            return Err(invalid("Tokenizer original26 case identity/order differs"));
        }
        let directory = evidence.join(name);
        std::fs::create_dir(&directory)?;
        let supplied = case["Files"]
            .as_array()
            .filter(|v| !v.is_empty() && v.len() <= 12)
            .ok_or_else(|| invalid("Tokenizer case file extent outside bounds"))?;
        let mut materialized = BTreeSet::new();
        let mut corrections = Vec::new();
        for path in supplied {
            let path = path
                .as_str()
                .ok_or_else(|| invalid("Tokenizer case input path is not text"))?;
            filename(path, name)?;
            if !consumed.insert(path) {
                return Err(invalid("Tokenizer input consumed twice"));
            }
            let row = identities
                .get(path)
                .ok_or_else(|| invalid("Tokenizer file identity missing"))?;
            let asset = engine.asset_path(&format!("{PREFIX}{path}"))?;
            let raw = read_locked(&asset, number(row, "Bytes")?, string(row, "Sha256")?)?;
            let bytes = materialize(&raw, row)?;
            let destination = row
                .get("MaterializedPath")
                .and_then(Value::as_str)
                .unwrap_or(path);
            let file = filename(destination, name)?;
            if !materialized.insert(file.to_owned()) {
                return Err(invalid("Tokenizer two inputs materialize the same file"));
            }
            write_new(&directory.join(file), &bytes)?;
            if row.get("Transformation").is_some() {
                corrections.push(json!({"Input":path,"InputSha256":row["Sha256"],"Output":destination,"Target":row["Target"],"Transformation":row["Transformation"]}));
            }
        }
        if !materialized.contains("tokenizer_config.json")
            || (name != "baichuan" && !materialized.contains("tokenizer.json"))
        {
            return Err(invalid("Tokenizer reference inputs incomplete"));
        }
        let vocabulary = directory.join("vocab.gguf");
        let identity = &case["Vocabulary"];
        let vocab_receipt = match string(identity, "Kind")? {
            "PinnedSourceArchive" => {
                let path = string(identity, "Path")?;
                if path != format!("models/ggml-vocab-{name}.gguf") {
                    return Err(invalid("Tokenizer source vocabulary path changed"));
                }
                let raw = read_locked(
                    &engine.source_path(path)?,
                    number(identity, "Bytes")?,
                    string(identity, "Sha256")?,
                )?;
                write_new(&vocabulary, &raw)?;
                json!({"PinnedSourceArchive":true,"Path":path,"Length":raw.len(),"Sha256":identity["Sha256"]})
            }
            "GeneratedVocabFixture" => {
                generated += 1;
                super::tokenizer_vocab::write(name, &directory, &vocabulary)?
            }
            _ => return Err(invalid("Unknown tokenizer vocabulary input kind")),
        };
        receipts.push(json!({"Name":name,"ReferenceFiles":materialized,"OriginalRepository":case["OriginalRepository"],"OriginalRevision":case["OriginalRevision"],"InputReconstruction":corrections,"Vocabulary":vocab_receipt}));
        result.push(Case {
            name: name.to_owned(),
            vocabulary,
            reference_directory: directory,
        });
    }
    if consumed.len() != files.len() || generated != 13 {
        return Err(invalid(
            "Tokenizer complete input coverage differs: expected122 files and13 generated fixtures",
        ));
    }
    Ok(Prepared {
        cases: result,
        receipt: json!({"AssetLockSha256":super::lora_inputs::hash_bytes(LOCK.as_bytes()),"VerifiedPublisherFiles":files.len(),"VerifiedPublisherBytes":total,"PinnedSourceVocabularies":13,"GeneratedVocabFixtures":generated,"SourceDocuments":documents,"MptCanonicalDependency":manifest["MptCanonicalDependency"],"Cases":receipts}),
    })
}
