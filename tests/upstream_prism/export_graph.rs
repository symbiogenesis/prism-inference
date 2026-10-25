//! Rust-owned graph filtering, structural ordering, deduplication and export.
use super::engine::Engine;
use super::gguf_model_data::{self, Model};
use super::model_session::{Client, ModelSession};
use super::{Result, invalid, result};
use serde::Deserialize;
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::io::Write as _;
use std::path::{Path, PathBuf};

const ID: &str = "tests/test-export-graph-ops.cpp";
const SOURCE_SHA: &str = "cfc871218eb04130cb0d6aa8241ff9bdb459c607a27a283701a85cc6325075f7";
const LOCAL_MODEL: &str = "tinyllamas/stories15M-q4_0.gguf";
#[derive(Default, Deserialize)]
#[serde(rename_all = "PascalCase", deny_unknown_fields)]
pub(super) struct Options {
    model: Option<PathBuf>,
    hf_repo: Option<String>,
    arguments: Vec<String>,
}
#[derive(Eq, PartialEq, Ord, PartialOrd)]
struct Input {
    kind: i32,
    ne: [i64; 4],
    nb: [u64; 4],
}
#[derive(Eq, PartialEq, Ord, PartialOrd)]
struct Key {
    op: i32,
    kind: i32,
    ne: [i64; 4],
    params: Vec<i32>,
    sources: Vec<Input>,
}
fn integer(value: &Value, name: &str) -> Result<i32> {
    i32::try_from(
        value[name]
            .as_i64()
            .ok_or_else(|| invalid(format!("graph export missing integer {name}")))?,
    )
    .map_err(Into::into)
}
fn unsigned(value: &Value, name: &str) -> Result<u64> {
    value[name]
        .as_u64()
        .ok_or_else(|| invalid(format!("graph export missing unsigned {name}")))
}
fn array_i64(value: &Value) -> Result<[i64; 4]> {
    let values = value
        .as_array()
        .filter(|v| v.len() == 4)
        .ok_or_else(|| invalid("graph dimensions must have4 entries"))?;
    Ok([
        values[0]
            .as_i64()
            .ok_or_else(|| invalid("invalid dimension"))?,
        values[1]
            .as_i64()
            .ok_or_else(|| invalid("invalid dimension"))?,
        values[2]
            .as_i64()
            .ok_or_else(|| invalid("invalid dimension"))?,
        values[3]
            .as_i64()
            .ok_or_else(|| invalid("invalid dimension"))?,
    ])
}
fn array_u64(value: &Value) -> Result<[u64; 4]> {
    let values = value
        .as_array()
        .filter(|v| v.len() == 4)
        .ok_or_else(|| invalid("graph strides must have4 entries"))?;
    Ok([
        values[0]
            .as_u64()
            .ok_or_else(|| invalid("invalid stride"))?,
        values[1]
            .as_u64()
            .ok_or_else(|| invalid("invalid stride"))?,
        values[2]
            .as_u64()
            .ok_or_else(|| invalid("invalid stride"))?,
        values[3]
            .as_u64()
            .ok_or_else(|| invalid("invalid stride"))?,
    ])
}
fn words(value: &Value) -> Result<Vec<i32>> {
    let hex = value
        .as_str()
        .filter(|s| s.len() % 8 == 0 && s.len() <= 4096)
        .ok_or_else(|| invalid("invalid graph op parameter bytes"))?;
    let raw = hex
        .as_bytes()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| -> Result<u8> {
            let digit = |byte: u8| match byte {
                b'0'..=b'9' => Ok(byte - b'0'),
                b'a'..=b'f' => Ok(byte - b'a' + 10),
                _ => Err(invalid("invalid op parameter hex")),
            };
            Ok(digit(pair[0])? * 16 + digit(pair[1])?)
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(raw
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&chunk| i32::from_le_bytes(chunk))
        .collect())
}
fn extract(graph: &Value, ignored: &[i32], tests: &mut BTreeMap<Key, String>) -> Result<Value> {
    if graph["Present"] != true {
        return Err(invalid("failed to reserve source graph"));
    }
    let nodes = graph["Nodes"]
        .as_array()
        .ok_or_else(|| invalid("reserved nodes absent"))?;
    let before = tests.len();
    let mut skipped = 0;
    for (index, node) in nodes.iter().enumerate() {
        if unsigned(node, "Index")? != u64::try_from(index)? {
            return Err(invalid("production graph order differs"));
        }
        let op = integer(node, "Op")?;
        if ignored.contains(&op) {
            skipped += 1;
            continue;
        }
        let mut sources = Vec::new();
        for source in node["Sources"]
            .as_array()
            .ok_or_else(|| invalid("node sources absent"))?
        {
            if source.is_null() {
                break;
            }
            sources.push(Input {
                kind: integer(source, "Type")?,
                ne: array_i64(&source["NE"])?,
                nb: array_u64(&source["NB"])?,
            });
        }
        let key = Key {
            op,
            kind: integer(node, "Type")?,
            ne: array_i64(&node["NE"])?,
            params: words(&node["OpParamsHex"])?,
            sources,
        };
        let name = node["Name"]
            .as_str()
            .ok_or_else(|| invalid("node name absent"))?;
        // std::set's comparator excludes the name, retaining the first node's.
        tests.entry(key).or_insert_with(|| name.to_owned());
    }
    Ok(json!({"TotalNodes":nodes.len(),"SkippedViews":skipped,"NewUniqueOps":tests.len()-before}))
}
fn serialize(tests: &BTreeMap<Key, String>) -> Result<String> {
    let mut text = String::new();
    for (test, name) in tests {
        write!(text, "{} {} ", test.op, test.kind)?;
        for n in test.ne {
            write!(text, "{n} ")?;
        }
        write!(text, "{} ", test.params.len())?;
        for n in &test.params {
            write!(text, "{n} ")?;
        }
        write!(text, "{} ", test.sources.len())?;
        for source in &test.sources {
            write!(text, "{} ", source.kind)?;
            for n in source.ne {
                write!(text, "{n} ")?;
            }
            for n in source.nb {
                write!(text, "{n} ")?;
            }
        }
        writeln!(text, "{}", if name.is_empty() { "-" } else { name })?;
    }
    Ok(text)
}
fn raw_string(bytes: &mut Vec<u8>, value: &[u8]) -> Result<()> {
    bytes.extend_from_slice(&u64::try_from(value.len())?.to_le_bytes());
    bytes.extend_from_slice(value);
    Ok(())
}
fn write_metadata(path: &Path, model: &Model) -> Result<()> {
    let mut bytes = b"GGUF".to_vec();
    bytes.extend_from_slice(&model.version.to_le_bytes());
    bytes.extend_from_slice(&u64::try_from(model.tensors.len())?.to_le_bytes());
    bytes.extend_from_slice(&u64::try_from(model.entries.len())?.to_le_bytes());
    for entry in &model.entries {
        raw_string(&mut bytes, &entry.key)?;
        bytes.extend_from_slice(&u32::try_from(entry.kind)?.to_le_bytes());
        bytes.extend_from_slice(&entry.payload);
    }
    for tensor in &model.tensors {
        raw_string(&mut bytes, &tensor.name)?;
        bytes.extend_from_slice(&tensor.n_dims.to_le_bytes());
        for &dimension in tensor
            .dimensions
            .iter()
            .take(usize::try_from(tensor.n_dims)?)
        {
            bytes.extend_from_slice(&u64::try_from(dimension)?.to_le_bytes());
        }
        bytes.extend_from_slice(&u32::try_from(tensor.kind)?.to_le_bytes());
        bytes.extend_from_slice(&tensor.offset.to_le_bytes());
    }
    let alignment =
        if let Some(entry) = model.entries.iter().find(|e| e.key == b"general.alignment") {
            if entry.kind != 4 {
                return Err(invalid("invalid GGUF alignment type"));
            }
            usize::try_from(u32::from_le_bytes(entry.payload.as_slice().try_into()?))?
        } else {
            32
        };
    if !alignment.is_power_of_two() || alignment > 1_048_576 {
        return Err(invalid("GGUF alignment exceeds bound"));
    }
    bytes.resize(
        bytes
            .len()
            .checked_add(alignment - 1)
            .ok_or_else(|| invalid("metadata padding overflow"))?
            & !(alignment - 1),
        0,
    );
    if bytes.len() > 64 * 1024 * 1024 {
        return Err(invalid("graph metadata exceeds64MiB"));
    }
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?;
    file.write_all(&bytes)?;
    Ok(())
}
fn call(client: &Client, value: Value) -> Result<Value> {
    Ok(client.call(0, value, None)?.value)
}
fn handle(value: &Value, name: &str) -> Result<u64> {
    unsigned(value, name)?
        .checked_sub(1)
        .map(|n| n + 1)
        .ok_or_else(|| invalid(format!("source null {name}")))
}
fn one(engine: &Engine, options: &Options, label: &str) -> Result<Value> {
    let Some(session) = ModelSession::open(engine, &format!("export-graph-{label}"))? else {
        return Err(invalid("production graph reservation API unavailable"));
    };
    let client = session.client();
    call(&client, json!({"Op":"common-init"}))?;
    call(&client, json!({"Op":"backend-init"}))?;
    let devices = call(&client, json!({"Op":"devices"}))?;
    let cpu = devices["Devices"]
        .as_array()
        .and_then(|devices| devices.iter().find(|d| d["Type"] == 0))
        .ok_or_else(|| invalid("source requires CPU device"))?;
    let cpu_index = unsigned(cpu, "Index")?;
    let mut metadata_receipt = Value::Null;
    let (model, context) = if let Some(repo_tag) = &options.hf_repo {
        let (repo, quant) = repo_tag
            .split_once(':')
            .unwrap_or((repo_tag.as_str(), "Q4_K_M"));
        let quant = if quant.is_empty() || quant == "latest" {
            "Q4_K_M"
        } else {
            quant
        };
        let mut provider = gguf_model_data::Frozen::open(engine)?;
        let cache = engine.evidence_path(&format!("export-graph-metadata-{label}"))?;
        let metadata = gguf_model_data::fetch_context(&mut provider, &cache, repo, quant)?
            .ok_or_else(|| invalid("failed to fetch source GGUF metadata"))?;
        let path = session.evidence_directory().join("model-metadata.gguf");
        write_metadata(&path, &metadata)?;
        metadata_receipt = json!({"Repo":repo,"Quant":quant,"Tensors":metadata.tensors.len(),"PrefixReads":provider.prefix_reads,"SourceCorrections":provider.corrections()});
        let value = call(
            &client,
            json!({"Op":"model-user-metadata","MetadataFile":path,"GpuLayers":0,"Devices":[cpu_index]}),
        )?;
        let model = handle(&value, "Model")?;
        let value = call(
            &client,
            json!({"Op":"context","Model":model,"Params":null,"Overrides":{}}),
        )?;
        (model, handle(&value, "Context")?)
    } else {
        let model_path = if let Some(path) = &options.model {
            path.clone()
        } else {
            engine.asset_path(LOCAL_MODEL)?
        };
        if !model_path.is_absolute() || !model_path.is_file() {
            return Err(invalid("export model must be existing absolute file"));
        }
        let mut arguments = vec![
            "test-export-graph-ops".to_owned(),
            "-m".to_owned(),
            model_path.to_string_lossy().into_owned(),
        ];
        arguments.extend(options.arguments.clone());
        let value = call(
            &client,
            json!({"Op":"params","Initial":{},"Arguments":arguments,"Example":"export-graph-ops"}),
        )?;
        if value["Parsed"] != true {
            return Err(invalid("source export arguments rejected"));
        }
        let params = handle(&value, "Handle")?;
        call(
            &client,
            json!({"Op":"params-update","Params":params,"Fields":{"Devices":[cpu_index],"FitParams":false,"GpuLayers":0,"Warmup":false}}),
        )?;
        let value = call(
            &client,
            json!({"Op":"common-model","Params":params,"ModelOnly":false}),
        )?;
        (handle(&value, "Model")?, handle(&value, "Context")?)
    };
    let limits = call(&client, json!({"Op":"context-limits","Context":context}))?;
    let ignored = ["NONE", "VIEW", "RESHAPE", "PERMUTE", "TRANSPOSE"]
        .iter()
        .map(|key| integer(&limits["Ops"], key))
        .collect::<Result<Vec<_>>>()?;
    let sequences = unsigned(&limits, "SeqMax")?;
    let count = unsigned(&limits, "Ctx")?.min(unsigned(&limits, "Ubatch")?);
    let mut tests = BTreeMap::new();
    let mut receipts = Vec::new();
    for (phase, tokens) in [("pp", count), ("tg", sequences)] {
        let graph = call(
            &client,
            json!({"Op":"context-graph-reserve","Context":context,"Tokens":tokens,"Sequences":sequences,"Outputs":tokens}),
        )?;
        receipts.push(json!({"Phase":phase,"Extraction":extract(&graph,&ignored,&mut tests)?}));
    }
    let text = serialize(&tests)?;
    let output = session.evidence_directory().join("tests.txt");
    // Match Windows text-mode std::ofstream's LF -> CRLF translation.
    std::fs::write(&output, text.replace('\n', "\r\n"))?;
    for (kind, id) in [("context", context), ("model", model)] {
        call(&client, json!({"Op":"release","Kind":kind,"Handle":id}))?;
    }
    call(&client, json!({"Op":"backend-free"}))?;
    session.finish()?;
    Ok(
        json!({"Branch":label,"Limits":limits,"Metadata":metadata_receipt,"Phases":receipts,"UniqueOps":tests.len(),"Output":output}),
    )
}
#[allow(
    clippy::unnecessary_wraps,
    reason = "Shared source registry expects a Result; source failures remain measured case results."
)]
pub(super) fn run(engine: &Engine) -> Result<Value> {
    let execute = (|| -> Result<Value> {
        let local = one(engine, &Options::default(), "local")?;
        let hf = one(
            engine,
            &Options {
                hf_repo: Some("ggml-org/Qwen3-0.6B-GGUF:Q8_0".into()),
                ..Options::default()
            },
            "hf",
        )?;
        Ok(
            json!({"SourceSha256":SOURCE_SHA,"OriginalNumericAssertions":0,"CompletionGuards":"production model/context/reserve success, graph observation extents and output creation","Branches":[local,hf]}),
        )
    })();
    Ok(match execute {
        Ok(value) => result(ID, true, value.to_string()),
        Err(error) => result(
            ID,
            false,
            format!("complete export source workload: {error}"),
        ),
    })
}
pub(super) fn diagnostic(engine: &Engine, value: &Value) -> Result<Value> {
    super::llama_grammar::exact_fields(value, &["Kind", "Options"])?;
    one(
        engine,
        &serde_json::from_value(value["Options"].clone())?,
        "diagnostic",
    )
}
#[cfg(test)]
mod tests {
    #[test]
    fn structural_key_retains_first_name_and_full_i32_parameters() -> super::Result<()> {
        let node = serde_json::json!({"Index":0,"Op":7,"Type":1,"NE":[4,2,1,1],"NB":[2,8,16,16],"OpParamsHex":"ffffffff00000080","Name":"first","Sources":[null]});
        let mut second = node.clone();
        second["Index"] = serde_json::json!(1);
        second["Name"] = serde_json::json!("second");
        let mut map = std::collections::BTreeMap::new();
        super::extract(
            &serde_json::json!({"Present":true,"Nodes":[node,second]}),
            &[],
            &mut map,
        )?;
        assert_eq!(map.len(), 1);
        assert_eq!(
            super::serialize(&map)?,
            "7 1 4 2 1 1 2 -1 -2147483648 0 first\n"
        );
        assert!(super::words(&serde_json::json!("zzzzzzzz")).is_err());
        Ok(())
    }
}
