//! Full metadata-helper and production quantization-selection source contracts.
//!
//! Pinned expected schema files remain external and unchanged. Frozen real HF
//! responses are independent inputs. No upstream C++ test helper is compiled.
//! MIT attribution: tools/prism/upstream/LICENSE.

use super::engine::Engine;
use super::gguf_model_data::{Frozen, Model};
use super::llama_grammar::{array, exact_fields, field, uint};
use super::{Result, invalid, result};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::fs;
use std::path::Path;

const META_ID: &str = "tests/test-gguf-model-data.cpp";
const QUANT_ID: &str = "tests/test-quant-type-selection.cpp";

use bonsai_prism_runtime::model::quantization_policy::{
    ModelDescriptor, Parameters, Policy, TYPE_COUNT, TensorDescriptor, default_type, type_name,
};

fn runtime_policy(model: &Value) -> Result<Policy> {
    exact_fields(
        model,
        &[
            "Architecture",
            "Embedding",
            "FeedForward",
            "Layers",
            "Heads",
            "KvHeads",
            "Experts",
            "HeadKey",
            "HeadValue",
        ],
    )?;
    let number = |name| {
        Ok::<_, Box<dyn std::error::Error + Send + Sync>>(u32::try_from(uint(model, name)?)?)
    };
    Policy::new(
        ModelDescriptor {
            architecture: text(model, "Architecture")?.to_owned(),
            embedding: number("Embedding")?,
            feed_forward: number("FeedForward")?,
            layers: number("Layers")?,
            heads: number("Heads")?,
            kv_heads: number("KvHeads")?,
            experts: number("Experts")?,
            head_key: number("HeadKey")?,
            head_value: number("HeadValue")?,
        },
        Parameters::default(),
    )
}

fn runtime_tensors(operation: &Value) -> Result<Vec<TensorDescriptor>> {
    let tensors = array(operation, "Tensors")?;
    if tensors.is_empty() || tensors.len() > 32768 {
        return Err(invalid("quantization tensor count exceeds bound"));
    }
    tensors
        .iter()
        .map(|tensor| {
            exact_fields(tensor, &["Name", "Dimensions"])?;
            let dimensions = array(tensor, "Dimensions")?;
            if dimensions.len() != 4 {
                return Err(invalid("quantization tensors require four dimensions"));
            }
            let mut shape = [0; 4];
            for (dimension, value) in shape.iter_mut().zip(dimensions) {
                *dimension = value
                    .as_u64()
                    .ok_or_else(|| invalid("quantization dimension must be unsigned"))?;
            }
            TensorDescriptor::new(text(tensor, "Name")?, shape)
        })
        .collect()
}

fn runtime_model(operation: &Value) -> Result<Value> {
    exact_fields(operation, &["Kind", "Model", "Tensors", "Actions"])?;
    let policy = runtime_policy(field(operation, "Model")?)?;
    let tensors = runtime_tensors(operation)?;
    let actions = array(operation, "Actions")?;
    if actions.len() > 128 {
        return Err(invalid("quantization action count exceeds bound"));
    }
    let mut returned = 0;
    let mut results = Vec::with_capacity(actions.len());
    for action in actions {
        match text(action, "Op")? {
            "allows" => {
                exact_fields(action, &["Op"])?;
                returned += tensors.len();
                results.push(json!({"Allowed":tensors.iter().map(|tensor|policy.allows(tensor)).collect::<Vec<_>>(),"Names":tensors.iter().map(TensorDescriptor::name).collect::<Vec<_>>()}));
            }
            "types" => {
                exact_fields(action, &["Op", "Ftype", "TensorIndices"])?;
                let ftype = u32::try_from(uint(action, "Ftype")?)?;
                if ftype >= 1024 {
                    return Err(invalid("quantization ftype exceeds bound"));
                }
                let indices = array(action, "TensorIndices")?;
                if indices.len() > tensors.len() {
                    return Err(invalid("quantization indices exceed tensor count"));
                }
                let selected = indices
                    .iter()
                    .map(|index| {
                        let index = usize::try_from(
                            index
                                .as_u64()
                                .ok_or_else(|| invalid("quantization index must be unsigned"))?,
                        )?;
                        let tensor = tensors
                            .get(index)
                            .ok_or_else(|| invalid("quantization index exceeds tensor count"))?;
                        if !policy.allows(tensor) {
                            return Err(invalid("quantization types require eligible tensors"));
                        }
                        Ok(tensor)
                    })
                    .collect::<Result<Vec<_>>>()?;
                let types = policy.compute(ftype, &selected)?;
                returned += selected.len();
                results.push(json!({"DefaultType":default_type(ftype),"Types":types}));
            }
            _ => return Err(invalid("unknown quantization action")),
        }
        if returned > 2 * 1024 * 1024 {
            return Err(invalid("quantization response exceeds bound"));
        }
    }
    Ok(json!({"Created":true,"Results":results}))
}

fn runtime_operation(operation: &Value) -> Result<Value> {
    if text(operation, "Kind")? == "quantization-model" {
        return runtime_model(operation);
    }
    exact_fields(operation, &["Kind", "Ftypes"])?;
    let ftypes = array(operation, "Ftypes")?;
    if ftypes.len() > 1024 {
        return Err(invalid("quantization ftype count exceeds bound"));
    }
    let defaults = ftypes
        .iter()
        .map(|ftype| {
            let ftype = u32::try_from(
                ftype
                    .as_u64()
                    .ok_or_else(|| invalid("quantization ftype must be unsigned"))?,
            )?;
            if ftype >= 1024 {
                return Err(invalid("quantization ftype exceeds bound"));
            }
            Ok(default_type(ftype))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(
        json!({"TypeCount":TYPE_COUNT,"FtypeGuessed":1024,"Types":(0..TYPE_COUNT).map(|kind|json!({"Type":kind,"Name":type_name(kind)})).collect::<Vec<_>>(),"Defaults":defaults}),
    )
}

pub(super) fn execute_runtime(request: &Value) -> Result<Option<Value>> {
    exact_fields(request, &["SchemaVersion", "Operations"])?;
    if uint(request, "SchemaVersion")? != 1 {
        return Err(invalid("quantization protocol version differs"));
    }
    let operations = array(request, "Operations")?;
    if operations.is_empty() || operations.len() > 128 {
        return Err(invalid("quantization operation count exceeds bound"));
    }
    if operations.iter().any(|operation| {
        !matches!(
            operation["Kind"].as_str(),
            Some("quantization-info" | "quantization-model")
        )
    }) {
        return Ok(None);
    }
    let results = operations
        .iter()
        .map(runtime_operation)
        .collect::<Result<Vec<_>>>()?;
    Ok(Some(json!({"SchemaVersion":1,"Results":results})))
}
// Original source model identity, quant request, existing schema and full extent.
const MODELS: [(&str, &str, &str, usize); 11] = [
    ("ggml-org/Qwen3-0.6B-GGUF", "Q8_0", "qwen3-0.6b", 1153),
    ("ggml-org/GLM-4.6V-GGUF", "Q8_0", "glm-4.6v", 3984),
    (
        "ggml-org/Step-3.5-Flash-GGUF",
        "Q4_K",
        "step-3.5-flash",
        2382,
    ),
    (
        "ggml-org/Qwen3-Coder-Next-GGUF",
        "Q8_0",
        "qwen3-coder-next",
        2070,
    ),
    ("ggml-org/Qwen3-14B-GGUF", "Q8_0", "qwen3-14b", 1837),
    (
        "ggml-org/NVIDIA-Nemotron-Nano-3-30B-A3B-GGUF",
        "Q8_0",
        "nemotron-nano-3-30b-a3b",
        3286,
    ),
    ("ggml-org/gpt-oss-120b-GGUF", "mxfp4", "gpt-oss-120b", 5529),
    ("ggml-org/gemma-3-4b-it-GGUF", "Q8_0", "gemma-3-4b-it", 1384),
    (
        "bartowski/Meta-Llama-3.1-70B-Instruct-GGUF",
        "Q4_K_M",
        "meta-llama-3.1-70b-instruct",
        3828,
    ),
    (
        "bartowski/deepseek-ai_DeepSeek-V3.1-GGUF",
        "IQ1_M",
        "deepseek-v3.1",
        3288,
    ),
    ("ggml-org/Qwen3.6-27B-GGUF", "Q8_0", "qwen3.6-27b", 2338),
];
// Exact source ftype table, with pinned public enum discriminants as input IDs.
const FTYPES: [(&str, u32); 34] = [
    ("F32", 0),
    ("F16", 1),
    ("BF16", 32),
    ("Q4_0", 2),
    ("Q4_1", 3),
    ("Q5_0", 8),
    ("Q5_1", 9),
    ("Q8_0", 7),
    ("Q2_K", 10),
    ("Q2_K_S", 21),
    ("Q3_K_S", 11),
    ("Q3_K_M", 12),
    ("Q3_K_L", 13),
    ("Q4_K_S", 14),
    ("Q4_K_M", 15),
    ("Q5_K_S", 16),
    ("Q5_K_M", 17),
    ("Q6_K", 18),
    ("IQ1_S", 24),
    ("IQ1_M", 31),
    ("IQ2_XXS", 19),
    ("IQ2_XS", 20),
    ("IQ2_S", 28),
    ("IQ2_M", 29),
    ("IQ3_XXS", 23),
    ("IQ3_XS", 22),
    ("IQ3_S", 26),
    ("IQ3_M", 27),
    ("IQ4_NL", 25),
    ("IQ4_XS", 30),
    ("TQ1_0", 36),
    ("TQ2_0", 37),
    ("MXFP4_MOE", 38),
    ("NVFP4", 39),
];

fn text<'a>(value: &'a Value, name: &str) -> Result<&'a str> {
    field(value, name)?
        .as_str()
        .ok_or_else(|| invalid(format!("quantization {name} must be text")))
}
fn digest(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .flat_map(|byte| [byte >> 4, byte & 15])
        .map(|n| char::from(b"0123456789abcdef"[usize::from(n)]))
        .collect()
}
fn input_summary(model: &Model) -> Value {
    let mut raw = Vec::new();
    raw.extend(model.version.to_le_bytes());
    for entry in &model.entries {
        raw.extend(&entry.key);
        raw.extend(entry.kind.to_le_bytes());
        raw.extend(&entry.payload);
    }
    for tensor in &model.tensors {
        raw.extend(&tensor.name);
        raw.extend(tensor.kind.to_le_bytes());
        raw.extend(tensor.n_dims.to_le_bytes());
        for dim in tensor.dimensions {
            raw.extend(dim.to_le_bytes());
        }
        raw.extend(tensor.offset.to_le_bytes());
    }
    json!({"ArchitectureBytes":model.architecture,"Layers":model.n_layer,"Tensors":model.tensors.len(),"SplitCount":model.n_split,"SplitTensorCount":model.n_split_tensors,"ParsedDescriptorSha256":digest(&raw)})
}
fn required(provider: &mut Frozen<'_>, cache: &Path, repo: &str, quant: &str) -> Result<Model> {
    provider.fetch(cache,repo,quant)?.ok_or_else(||invalid(format!("required immutable GGUF metadata failed: {repo}/{quant}; source network skip cannot count as coverage")))
}

pub(super) fn run_metadata(engine: &Engine) -> Result<Value> {
    let mut provider = Frozen::open(engine)?;
    let cache = engine.evidence_path("metadata-helper-cache")?;
    let mut failures = Vec::new();
    let mut count = 0;
    let mut check = |name: &str, passed: bool| {
        count += 1;
        if !passed {
            failures.push(name.to_owned());
        }
    };
    let q = required(&mut provider, &cache, MODELS[0].0, MODELS[0].1)?;
    check("qwen architecture", q.architecture == b"qwen3");
    check("qwen layers", q.n_layer == 28);
    check("qwen embedding", q.n_embd == 1024);
    check("qwen heads", q.n_head == 16);
    check("qwen KV heads", q.n_head_kv == 8);
    check("qwen experts", q.n_expert == 0);
    check("qwen vocabulary", q.n_vocab == 151_936);
    check("qwen tensor count", q.tensors.len() == 311);
    for name in [
        b"blk.0.attn_q.weight".as_slice(),
        b"token_embd.weight".as_slice(),
        b"output_norm.weight".as_slice(),
    ] {
        check(
            "qwen tensor name",
            q.tensors.iter().any(|tensor| tensor.name == name),
        );
    }
    let embedding = q
        .tensors
        .iter()
        .find(|tensor| tensor.name == b"token_embd.weight");
    check(
        "qwen embedding dimension",
        embedding.is_some_and(|tensor| tensor.dimensions[0] == 1024),
    );
    check(
        "qwen embedding rank",
        embedding.is_some_and(|tensor| tensor.n_dims == 2),
    );
    let prefix_before = provider.prefix_reads;
    let cached = provider.fetch(&cache, MODELS[0].0, MODELS[0].1)?;
    check("cached fetch exists", cached.is_some());
    check(
        "cached fetch tensor count",
        cached
            .as_ref()
            .is_some_and(|model| model.tensors.len() == q.tensors.len()),
    );
    let cached_prefix_reads = provider.prefix_reads - prefix_before;
    let glm = required(&mut provider, &cache, MODELS[1].0, "Q8_0")?;
    check("glm architecture", glm.architecture == b"glm4moe");
    check("glm layers", glm.n_layer == 46);
    check("glm embedding", glm.n_embd == 4096);
    check("glm heads", glm.n_head == 96);
    check("glm KV heads", glm.n_head_kv == 8);
    check("glm experts", glm.n_expert == 128);
    check("glm vocabulary", glm.n_vocab == 151_552);
    check("glm tensor count", glm.tensors.len() == 780);
    let step = required(&mut provider, &cache, MODELS[2].0, MODELS[2].1)?;
    check("step architecture", step.architecture == b"step35");
    check("step layers", step.n_layer == 45);
    check("step embedding", step.n_embd == 4096);
    check("step feedforward", step.n_ff == 11264);
    check("step heads first array value", step.n_head == 64);
    check("step KV heads first array value", step.n_head_kv == 8);
    check("step experts", step.n_expert == 288);
    check("step key head dimension", step.n_embd_head_k == 128);
    check("step value head dimension", step.n_embd_head_v == 128);
    check("step vocabulary", step.n_vocab == 128_896);
    check("step tensor count", step.tensors.len() == 754);
    let mut output = result(
        META_ID,
        count == 34 && failures.is_empty(),
        format!(
            "{count}/34 original helper predicates; failures={failures:?}; target-independent safeRust helper rewrite, no production engine coverage claimed"
        ),
    );
    output["CoverageKind"] = json!("TargetIndependentTestHelper");
    output["CachedFetchPrefixReads"] = json!(cached_prefix_reads);
    output["Inputs"] = json!([input_summary(&q), input_summary(&glm), input_summary(&step)]);
    output["SourceCorrections"] = json!(provider.corrections());
    Ok(output)
}

fn exchange(engine: &Engine, operation: &Value) -> Result<Option<Value>> {
    let Some(response) = engine.execute_common(
        "quantization-selection",
        &json!({"SchemaVersion":1,"Operations":[operation]}),
    )?
    else {
        return Ok(None);
    };
    exact_fields(&response, &["SchemaVersion", "Results"])?;
    let values = array(&response, "Results")?;
    if uint(&response, "SchemaVersion")? != 1 || values.len() != 1 {
        return Err(invalid("quantization response framing invalid"));
    }
    Ok(Some(values[0].clone()))
}
struct Inventory {
    names: Vec<Option<String>>,
    defaults: Vec<u32>,
    count: u32,
}
impl Inventory {
    fn read(engine: &Engine) -> Result<Option<Self>> {
        let Some(value) = exchange(
            engine,
            &json!({"Kind":"quantization-info","Ftypes":(0..1024).collect::<Vec<u32>>()}),
        )?
        else {
            return Ok(None);
        };
        exact_fields(&value, &["TypeCount", "FtypeGuessed", "Types", "Defaults"])?;
        let count = u32::try_from(uint(&value, "TypeCount")?)?;
        let types = array(&value, "Types")?;
        let defaults = array(&value, "Defaults")?;
        if count > 4096
            || types.len() != usize::try_from(count)?
            || uint(&value, "FtypeGuessed")? != 1024
            || defaults.len() != 1024
        {
            return Err(invalid("quantization inventory extent invalid"));
        }
        let mut names = Vec::new();
        for (index, kind) in types.iter().enumerate() {
            exact_fields(kind, &["Type", "Name"])?;
            if uint(kind, "Type")? != u64::try_from(index)? {
                return Err(invalid("quantization type inventory order invalid"));
            }
            names.push(if kind["Name"].is_null() {
                None
            } else {
                Some(text(kind, "Name")?.to_owned())
            });
        }
        let defaults = defaults
            .iter()
            .map(|value| {
                u32::try_from(
                    value
                        .as_u64()
                        .ok_or_else(|| invalid("quantization default must be unsigned"))?,
                )
                .map_err(Into::into)
            })
            .collect::<Result<Vec<_>>>()?;
        if defaults.iter().any(|&value| value > count) {
            return Err(invalid("quantization default outside type enum"));
        }
        Ok(Some(Self {
            names,
            defaults,
            count,
        }))
    }
    fn type_id(&self, name: &str) -> Result<u32> {
        self.names
            .iter()
            .position(|value| value.as_deref() == Some(name))
            .ok_or_else(|| invalid(format!("unknown source GGML type name {name}")))
            .and_then(|value| u32::try_from(value).map_err(Into::into))
    }
    fn name(&self, id: u32) -> Result<&str> {
        self.names
            .get(usize::try_from(id)?)
            .and_then(Option::as_deref)
            .ok_or_else(|| invalid("quantization type name missing"))
    }
}
struct Section {
    ftype: u32,
    default: u32,
    overrides: Vec<(String, u32)>,
}
fn windows_text(bytes: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == 26 {
            break;
        }
        if bytes[i] == 13 && bytes.get(i + 1) == Some(&10) {
            i += 1;
        }
        out.push(bytes[i]);
        i += 1;
    }
    out
}
fn sections(bytes: &[u8], inventory: &Inventory) -> Result<Vec<Section>> {
    let bytes = windows_text(bytes);
    let text = std::str::from_utf8(&bytes)?;
    let mut sections: Vec<Section> = Vec::new();
    for line in text.split_terminator('\n') {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if line.starts_with('[') {
            let close = line
                .find(']')
                .ok_or_else(|| invalid("schema section lacks closing bracket"))?;
            let ftype = FTYPES
                .iter()
                .find(|(name, _)| *name == &line[1..close])
                .ok_or_else(|| invalid("schema ftype name unknown"))?
                .1;
            let default = inventory.type_id(line[close + 1..].trim_start_matches(' '))?;
            sections.push(Section {
                ftype,
                default,
                overrides: Vec::new(),
            });
        } else {
            let section = sections
                .last_mut()
                .ok_or_else(|| invalid("schema tensor appears before section"))?;
            let space = line
                .rfind(' ')
                .ok_or_else(|| invalid("schema tensor line lacks space"))?;
            section.overrides.push((
                line[..space].to_owned(),
                inventory.type_id(&line[space + 1..])?,
            ));
        }
    }
    Ok(sections)
}
fn model_request(model: &Model, actions: &Value) -> Result<Value> {
    let architecture = std::str::from_utf8(&model.architecture)?;
    let tensors = model
        .tensors
        .iter()
        .map(|tensor| {
            Ok(json!({"Name":std::str::from_utf8(&tensor.name)?,"Dimensions":tensor.dimensions}))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(
        json!({"Kind":"quantization-model","Model":{"Architecture":architecture,"Embedding":model.n_embd,"FeedForward":model.n_ff,"Layers":model.n_layer,"Heads":model.n_head,"KvHeads":model.n_head_kv,"Experts":model.n_expert,"HeadKey":model.n_embd_head_k,"HeadValue":model.n_embd_head_v},"Tensors":tensors,"Actions":actions}),
    )
}
fn model_results(
    engine: &Engine,
    model: &Model,
    actions: &Value,
    count: usize,
) -> Result<Vec<Value>> {
    let response = exchange(engine, &model_request(model, actions)?)?
        .ok_or_else(|| invalid("production quantization API disappeared"))?;
    exact_fields(&response, &["Created", "Results"])?;
    if field(&response, "Created")?.as_bool() != Some(true) {
        return Err(invalid("production metadata model creation failed"));
    }
    let values = array(&response, "Results")?;
    if values.len() != count {
        return Err(invalid("quantization action/result extent mismatch"));
    }
    Ok(values.to_vec())
}
fn layer(name: &str) -> i32 {
    let Some(value) = name.strip_prefix("blk.") else {
        return -1;
    };
    let value = value.trim_start_matches(|c: char| c.is_ascii_whitespace());
    let count = value
        .bytes()
        .enumerate()
        .take_while(|(i, c)| c.is_ascii_digit() || (*i == 0 && matches!(c, b'+' | b'-')))
        .count();
    value[..count].parse().unwrap_or(-1)
}
fn selected(engine: &Engine, model: &Model) -> Result<(Vec<String>, Vec<usize>)> {
    let values = model_results(engine, model, &json!([{"Op":"allows"}]), 1)?;
    exact_fields(&values[0], &["Allowed", "Names"])?;
    let allowed = array(&values[0], "Allowed")?;
    let names = array(&values[0], "Names")?;
    if allowed.len() != model.tensors.len() || names.len() != allowed.len() {
        return Err(invalid("quantizability response extent mismatch"));
    }
    let names = names
        .iter()
        .map(|value| {
            value
                .as_str()
                .map(str::to_owned)
                .ok_or_else(|| invalid("production tensor name must be text"))
        })
        .collect::<Result<Vec<_>>>()?;
    let mut selected = Vec::new();
    for (index, value) in allowed.iter().enumerate() {
        if value
            .as_bool()
            .ok_or_else(|| invalid("quantizability must be boolean"))?
        {
            selected.push(index);
        }
    }
    selected.sort_by(|&a, &b| {
        layer(&names[a])
            .cmp(&layer(&names[b]))
            .then_with(|| names[a].cmp(&names[b]))
    });
    Ok((names, selected))
}
fn assignments(value: &Value, indices: usize, inventory: &Inventory) -> Result<(u32, Vec<u32>)> {
    exact_fields(value, &["DefaultType", "Types"])?;
    let default = u32::try_from(uint(value, "DefaultType")?)?;
    let types = array(value, "Types")?;
    if types.len() != indices {
        return Err(invalid("quantization assignment extent mismatch"));
    }
    let types = types
        .iter()
        .map(|value| {
            u32::try_from(
                value
                    .as_u64()
                    .ok_or_else(|| invalid("assigned type must be unsigned"))?,
            )
            .map_err(Into::into)
        })
        .collect::<Result<Vec<_>>>()?;
    if default >= inventory.count || types.iter().any(|&value| value >= inventory.count) {
        return Err(invalid("assigned quantization type outside enum"));
    }
    Ok((default, types))
}

pub(super) fn run_quantization(engine: &Engine) -> Result<Value> {
    let Some(inventory) = Inventory::read(engine)? else {
        return Ok(result(QUANT_ID,false,"unsupported production metadata-model quantization APIs; all11realmodels and363expectedsections required".into()));
    };
    let mut provider = Frozen::open(engine)?;
    let cache = engine.evidence_path("quantization-metadata-cache")?;
    let mut total = 0;
    let mut failed = Vec::new();
    let mut model_rows = Vec::new();
    for &(repo, quant, file, expected_overrides) in &MODELS {
        let model = required(&mut provider, &cache, repo, quant)?;
        let schema_path = engine.source_path(&format!("tests/snapshots/{file}.schema"))?;
        let schema = fs::read(schema_path)?;
        let sections = sections(&schema, &inventory)?;
        if sections.len() != 33
            || sections
                .iter()
                .map(|section| section.overrides.len())
                .sum::<usize>()
                != expected_overrides
        {
            return Err(invalid(
                "complete pinned schema section/override extent mismatch",
            ));
        }
        let (names, indices) = selected(engine, &model)?;
        let actions = sections
            .iter()
            .map(|section| json!({"Op":"types","Ftype":section.ftype,"TensorIndices":indices}))
            .collect::<Vec<_>>();
        let values = model_results(engine, &model, &json!(actions), sections.len())?;
        let mut predicates = 0;
        let mut bad = 0;
        for (section, value) in sections.iter().zip(values) {
            total += 1;
            let (default, types) = assignments(&value, indices.len(), &inventory)?;
            predicates += 1;
            let mut okay = default == section.default;
            if okay {
                let mut overrides = BTreeMap::new();
                for (name, kind) in &section.overrides {
                    overrides.entry(name.as_str()).or_insert(*kind);
                }
                let mut found = 0;
                for (&index, actual) in indices.iter().zip(types) {
                    let expected = if let Some(&kind) = overrides.get(names[index].as_str()) {
                        found += 1;
                        kind
                    } else {
                        section.default
                    };
                    predicates += 1;
                    okay &= actual == expected;
                }
                predicates += 1;
                okay &= found == section.overrides.len();
            }
            if !okay {
                bad += 1;
                failed.push(format!("{file} ftype{}", section.ftype));
            }
        }
        model_rows.push(json!({"Repo":repo,"Quant":quant,"Schema":format!("{file}.schema"),"SchemaSha256":digest(&schema),"Sections":33,"Overrides":expected_overrides,"QuantizableTensors":indices.len(),"Predicates":predicates,"FailedSections":bad,"Input":input_summary(&model)}));
    }
    let mut output = result(
        QUANT_ID,
        total == 363 && failed.is_empty(),
        format!(
            "{total}/363 complete source sections across all11models; failures={failed:?}; frozen realheaders; schema expectations unchanged"
        ),
    );
    output["Models"] = json!(model_rows);
    output["SourceCorrections"] = json!({"InputProviders":provider.corrections(),"SchemaPath":"Pinned Nemotron source builds nonexistent nvidia-nemotron-nano-3-30b-a3b.schema; explicit fixture repair uses untouched nemotron-nano-3-30b-a3b.schema"});
    Ok(output)
}

fn model_name(repo: &str) -> String {
    let value = repo.split_once('/').map_or(repo, |(_, name)| name);
    let value = value.strip_suffix("-GGUF").unwrap_or(value);
    value
        .split_once('_')
        .map_or(value, |(_, name)| name)
        .to_owned()
}
/// Optional original generator. Invoke only explicitly in CI with confirmation.
/// It writes requested external outputs and never updates expected source files.
pub(super) fn generate(engine: &Engine, directory: &Path, confirmed: bool) -> Result<Value> {
    if !confirmed || !directory.is_absolute() {
        return Err(invalid(
            "source --generate requires explicit confirmation and absolute output directory",
        ));
    }
    let inventory = Inventory::read(engine)?
        .ok_or_else(|| invalid("production quantization generator API unavailable"))?;
    let mut provider = Frozen::open(engine)?;
    let cache = engine.evidence_path("quantization-generate-cache")?;
    fs::create_dir_all(directory)?;
    let mut written = Vec::new();
    for &(repo, quant, file, _) in &MODELS {
        let model = required(&mut provider, &cache, repo, quant)?;
        let (names, indices) = selected(engine, &model)?;
        let ftypes = (0..1024)
            .filter(|&id| {
                inventory.defaults[id] != inventory.count
                    && FTYPES
                        .iter()
                        .any(|(_, value)| usize::try_from(*value).ok() == Some(id))
            })
            .collect::<Vec<_>>();
        let actions = ftypes
            .iter()
            .map(|ftype| json!({"Op":"types","Ftype":ftype,"TensorIndices":indices}))
            .collect::<Vec<_>>();
        let values = model_results(engine, &model, &json!(actions), ftypes.len())?;
        let mut output = format!(
            "# Model: {}\n# n_embd={}, n_ff={}, n_vocab={}, n_layer={}, n_head={}, n_head_kv={}",
            model_name(repo),
            model.n_embd,
            model.n_ff,
            model.n_vocab,
            model.n_layer,
            model.n_head,
            model.n_head_kv
        );
        if model.n_expert > 0 {
            write!(output, ", n_expert={}", model.n_expert)?;
        }
        output.push('\n');
        for (id, value) in ftypes.into_iter().zip(values) {
            let name = FTYPES
                .iter()
                .find(|(_, value)| usize::try_from(*value).ok() == Some(id))
                .ok_or_else(|| invalid("generator ftype lookup missing"))?
                .0;
            let (default, types) = assignments(&value, indices.len(), &inventory)?;
            write!(output, "\n[{name}] {}\n", inventory.name(default)?)?;
            for (&index, kind) in indices.iter().zip(types) {
                if kind != default {
                    writeln!(output, "{} {}", names[index], inventory.name(kind)?)?;
                }
            }
        }
        let path = directory.join(format!("{file}.schema"));
        fs::write(&path, output.replace('\n', "\r\n"))?;
        written.push(path);
    }
    Ok(json!({"Written":written,"SourceCorrections":provider.corrections(),"DiagnosticOnly":true}))
}
