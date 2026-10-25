//! Rust orchestration of every corpus in the original external repository loop.
//! Repository bytes are immutable external inputs; no C++ test executable runs.
use super::engine::Engine;
use super::lora_inputs::digest;
use super::{Result, invalid, result};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
const ID: &str = "tests/test-tokenizers-repo.sh";
fn relative(value: &Value) -> Result<&str> {
    let name = value
        .as_str()
        .filter(|s| {
            !s.is_empty() && !s.contains("..") && !s.contains('\\') && !Path::new(s).is_absolute()
        })
        .ok_or_else(|| invalid("invalid tokenizer repository relative path"))?;
    Ok(name)
}
#[allow(
    clippy::unnecessary_wraps,
    reason = "all source ports share a Result interface with measured failure records"
)]
#[allow(
    clippy::case_sensitive_file_extension_comparisons,
    reason = "the source Bash glob selects only literal lowercase .gguf"
)]
pub(super) fn run(engine: &Engine) -> Result<Value> {
    let outcome = (|| -> Result<Value> {
        let lock: Value = serde_json::from_str(include_str!("tokenizer-repository-assets.json"))?;
        if lock["Revision"] != "cee4b7a68518c579bb758fca46bb149cd109348e" {
            return Err(invalid("external corpus revision differs"));
        }
        let files = lock["Files"]
            .as_array()
            .filter(|v| v.len() == 18)
            .ok_or_else(|| invalid("external corpus fixture lock incomplete"))?;
        let mut paths = BTreeMap::<String, PathBuf>::new();
        for entry in files {
            let name = relative(&entry["Path"])?;
            let path = engine.asset_path(&format!("tokenizer-repository/{name}"))?;
            if Some(path.metadata()?.len()) != entry["Bytes"].as_u64() {
                return Err(invalid(format!(
                    "external corpus input length differs: {name}"
                )));
            }
            let (_, sha) = digest(&path)?;
            if entry["Sha256"] != sha {
                return Err(invalid(format!(
                    "external corpus input hash differs: {name}"
                )));
            }
            if paths.insert(name.into(), path).is_some() {
                return Err(invalid("external corpus path repeated"));
            }
        }
        let mut receipts = Vec::new();
        let mut ignored = Vec::new();
        let mut predicates = 0_u64;
        for (name, path) in &paths {
            if !name.ends_with(".gguf") {
                continue;
            }
            let (Some(input), Some(output)) = (
                paths.get(&format!("{name}.inp")),
                paths.get(&format!("{name}.out")),
            ) else {
                ignored.push(name.clone());
                continue;
            };
            let corpus = match super::tokenizer::check_corpus(
                engine,
                name,
                path,
                &fs::read(input)?,
                &fs::read(output)?,
            ) {
                Ok(value) => value,
                Err(error) => {
                    json!({"Name":name,"Passed":false,"SourcePredicates":0,"Completed":false,"Error":error.to_string()})
                }
            };
            predicates = predicates
                .checked_add(
                    corpus["SourcePredicates"]
                        .as_u64()
                        .ok_or_else(|| invalid("corpus predicate count missing"))?,
                )
                .ok_or_else(|| invalid("corpus predicate count overflow"))?;
            receipts.push(corpus);
        }
        if receipts.len() != 6 || !ignored.is_empty() {
            return Err(invalid("pinned external repository enumeration differs"));
        }
        Ok(
            json!({"Revision":lock["Revision"],"Repository":lock["Repository"],"InputSelection":lock["InputSelection"],"CorpusCount":receipts.len(),"SourceTokenPredicates":predicates,"Corpora":receipts,"IgnoredMissingCorpus":ignored,"TestExecutable":"safe-rust-tokenizer-corpus","SourceCorrection":"Immutable repository asset provisioning replaces mutable git clone/pull; every triple in the compatible publisher snapshot is retained","PlatformExclusions":["s390x endian conversion branch does not apply to Windows ARM64"]}),
        )
    })();
    match outcome {
        Ok(value) => {
            let passed = value["Corpora"]
                .as_array()
                .is_some_and(|rows| rows.iter().all(|v| v["Passed"] == true));
            Ok(result(ID, passed, value.to_string()))
        }
        Err(error) => Ok(result(
            ID,
            false,
            format!("complete external tokenizer repository contract: {error}"),
        )),
    }
}
