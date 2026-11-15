//! Independent metadata quantization policy observations for upstream conformance.
//!
//! This module derives tensor quantization eligibility and per-tensor storage
//! types solely from metadata descriptors (`quantization-info` and
//! `quantization-model`) without calling `bonsai_prism_runtime`, any native
//! library, or expected-answer fixture files.

use serde::Deserialize;
use serde_json::{Value, json};

use super::{Result, invalid};

/// Upstream conformance test route identifier.
pub(super) const ID: &str = "tests/test-quant-type-selection.cpp";

const SOURCE_SHA256: &str = "9a4879f21341efb4adb14c0f08c6fa414735f5a8132ed22181de41efecaa9cb4";
const TYPE_COUNT: u32 = 144;
const FTYPE_GUESSED: u32 = 1024;
const MAX_FTYPES: usize = 1024;
const MAX_OPERATIONS: usize = 128;
const MAX_ARCH_BYTES: usize = 128;
const MAX_MODEL_PARAM: u32 = 1024 * 1024;
const MAX_LAYERS: u32 = 512;
const MAX_TENSORS: usize = 32768;
const MAX_TENSOR_NAME_INPUT: usize = 1024;
const MAX_TENSOR_NAME_NATIVE: usize = 63;
const MAX_ACTIONS: usize = 128;
const MAX_RETURNED: usize = 2 * 1024 * 1024;

#[derive(Deserialize)]
#[serde(rename_all = "PascalCase", deny_unknown_fields)]
struct Request {
    schema_version: u32,
    operations: Vec<Operation>,
}

#[derive(Deserialize)]
#[serde(tag = "Kind", deny_unknown_fields)]
enum Operation {
    #[serde(rename = "quantization-info")]
    Info {
        #[serde(rename = "Ftypes")]
        ftypes: Vec<u32>,
    },
    #[serde(rename = "quantization-model")]
    Model {
        #[serde(rename = "Model")]
        model: ModelDesc,
        #[serde(rename = "Tensors")]
        tensors: Vec<TensorDesc>,
        #[serde(rename = "Actions")]
        actions: Vec<Action>,
    },
}

#[derive(Deserialize)]
#[serde(rename_all = "PascalCase", deny_unknown_fields)]
struct ModelDesc {
    architecture: String,
    embedding: u32,
    feed_forward: u32,
    layers: u32,
    heads: u32,
    kv_heads: u32,
    experts: u32,
    head_key: u32,
    head_value: u32,
}

#[derive(Deserialize)]
#[serde(rename_all = "PascalCase", deny_unknown_fields)]
struct TensorDesc {
    name: String,
    dimensions: [u64; 4],
}

#[derive(Deserialize)]
#[serde(tag = "Op", deny_unknown_fields)]
enum Action {
    #[serde(rename = "allows")]
    Allows,
    #[serde(rename = "types")]
    Types {
        #[serde(rename = "Ftype")]
        ftype: u32,
        #[serde(rename = "TensorIndices")]
        tensor_indices: Vec<usize>,
    },
}

struct ValidatedModel<'a> {
    architecture: &'a str,
    layers: usize,
    heads: u32,
    kv_heads: u32,
    experts: u32,
}

impl ValidatedModel<'_> {
    const fn gqa(&self) -> u32 {
        match self.heads.checked_div(self.kv_heads) {
            Some(ratio) => ratio,
            None => 0,
        }
    }
}

struct ValidatedTensor {
    name: String,
    dimensions: [u64; 4],
}

enum ValidatedOp<'a> {
    Info(&'a [u32]),
    Model {
        model: ValidatedModel<'a>,
        tensors: Vec<ValidatedTensor>,
        actions: &'a [Action],
    },
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Category {
    TokenEmbedding,
    AttentionQ,
    AttentionV,
    AttentionK,
    AttentionQkv,
    AttentionKvB,
    AttentionOutput,
    FfnUp,
    FfnGate,
    FfnDown,
    Output,
    Other,
}

impl Category {
    const fn is_attention_v(self) -> bool {
        matches!(
            self,
            Self::AttentionV | Self::AttentionQkv | Self::AttentionKvB
        )
    }
}

#[derive(Default)]
struct Counters {
    attention_count: usize,
    attention: usize,
    down: usize,
    gate: usize,
    up: usize,
    tied_embeddings: bool,
}

const fn type_name(kind: u32) -> Option<&'static str> {
    Some(match kind {
        0 => "f32",
        1 => "f16",
        2 => "q4_0",
        3 => "q4_1",
        4 | 5 => "DEPRECATED",
        6 => "q5_0",
        7 => "q5_1",
        8 => "q8_0",
        9 => "q8_1",
        10 => "q2_K",
        11 => "q3_K",
        12 => "q4_K",
        13 => "q5_K",
        14 => "q6_K",
        15 => "q8_K",
        16 => "iq2_xxs",
        17 => "iq2_xs",
        18 => "iq3_xxs",
        19 => "iq1_s",
        20 => "iq4_nl",
        21 => "iq3_s",
        22 => "iq2_s",
        23 => "iq4_xs",
        24 => "i8",
        25 => "i16",
        26 => "i32",
        27 => "i64",
        28 => "f64",
        29 => "iq1_m",
        30 => "bf16",
        31 => "TYPE_Q4_0_4_4 REMOVED, use Q4_0 with runtime repacking",
        32 => "TYPE_Q4_0_4_8 REMOVED, use Q4_0 with runtime repacking",
        33 => "TYPE_Q4_0_8_8 REMOVED, use Q4_0 with runtime repacking",
        34 => "tq1_0",
        35 => "tq2_0",
        36 => "TYPE_IQ4_NL_4_4 REMOVED, use IQ4_NL with runtime repacking",
        37 => "TYPE_IQ4_NL_4_8 REMOVED, use IQ4_NL with runtime repacking",
        38 => "TYPE_IQ4_NL_8_8 REMOVED, use IQ4_NL with runtime repacking",
        39 => "mxfp4",
        40 => "nvfp4",
        41 => "q1_0",
        42 => "q2_0",
        142 => "pq2_0",
        143 => "ptq1_0",
        _ => return None,
    })
}

const fn type_block(kind: u32) -> Option<u64> {
    Some(match kind {
        0 | 1 | 24..=28 | 30 => 1,
        2 | 3 | 6..=9 | 20 | 39 => 32,
        40 | 42 => 64,
        41 | 142 | 143 => 128,
        10..=19 | 21..=23 | 29 | 34 | 35 => 256,
        _ => return None,
    })
}

const fn default_type(ftype: u32) -> u32 {
    match ftype {
        0 => 0,
        1 => 1,
        2 => 2,
        3 => 3,
        7 => 8,
        8 => 6,
        9 => 7,
        10 | 21 => 10,
        11..=13 => 11,
        14 | 15 => 12,
        16 | 17 => 13,
        18 => 14,
        19 => 16,
        20 | 28 => 17,
        22 | 26 | 27 => 21,
        23 => 18,
        24 => 19,
        25 => 20,
        29 => 22,
        30 => 23,
        31 => 29,
        32 => 30,
        36 => 34,
        37 => 35,
        38 => 39,
        40 => 41,
        41 => 42,
        141 => 142,
        143 => 143,
        _ => TYPE_COUNT,
    }
}

const fn more_bits(layer: usize, layers: usize) -> bool {
    layer < layers / 8 || layer >= 7 * layers / 8 || (layer - layers / 8) % 3 == 2
}

const fn low_iq(ftype: u32) -> bool {
    matches!(ftype, 19 | 20 | 24 | 28 | 29 | 31)
}

const fn fallback_candidate(kind: u32) -> Option<u32> {
    Some(match kind {
        16..=19 | 21..=23 | 29 => 20,
        10 | 11 | 34 | 35 | 42 | 142 | 143 => 2,
        12 => 6,
        13 => 7,
        14 => 8,
        _ => return None,
    })
}

const EXCLUSIONS: &[&str] = &[
    "_norm.weight",
    "ffn_gate_inp.weight",
    "ffn_gate_tid2eid.weight",
    "altup",
    "laurel",
    "per_layer_model_proj",
    "ssm_conv1d",
    "shortconv.conv.weight",
    "indexer.k_proj.weight",
    "indexer.q_proj.weight",
    "time_mix_first.weight",
    "time_mix_w0.weight",
    "time_mix_w1.weight",
    "time_mix_w2.weight",
    "time_mix_v0.weight",
    "time_mix_v1.weight",
    "time_mix_v2.weight",
    "time_mix_a0.weight",
    "time_mix_a1.weight",
    "time_mix_a2.weight",
    "time_mix_g1.weight",
    "time_mix_g2.weight",
    "time_mix_decay_w1.weight",
    "time_mix_decay_w2.weight",
    "time_mix_lerp_fused.weight",
    "attn_rel_b.weight",
    ".position_embd",
    "sam.pos_embd",
    "sam.neck.",
    "sam.net_",
    ".rel_pos",
    ".patch_embd",
    ".patch_merger",
    "a.rvq.codebook",
    "mm.a.code_embd",
];

const ARCHITECTURES: &str = "clip llama llama4 deci falcon grok gpt2 gptj gptneox mpt baichuan starcoder refact bert modern-bert nomic-bert nomic-bert-moe neo-bert jina-bert-v2 jina-bert-v3 eurobert bloom stablelm qwen qwen2 qwen2moe qwen2vl qwen3 qwen3moe qwen3next qwen3vl qwen3vlmoe qwen35 qwen35moe dspark phi2 phi3 phimoe plamo plamo2 plamo3 codeshell orion internlm2 minicpm minicpm3 gemma gemma2 gemma3 gemma3n gemma4 gemma4-assistant gemma-embedding starcoder2 mamba mamba2 jamba falcon-h1 xverse command-r cohere2 cohere2moe dbrx olmo olmo2 olmoe muse-glimmer openelm arctic deepseek deepseek2 deepseek2-ocr deepseek32 deepseek4 chatglm glm4 glm4moe glm-dsa bitnet t5 t5encoder jais jais2 nemotron nemotron_h nemotron_h_moe exaone exaone4 exaone-moe rwkv6 rwkv6qwen2 rwkv7 arwkv7 granite granitemoe granitehybrid graniteswitch granite_swa chameleon wavtokenizer-dec plm bailingmoe bailingmoe2 bailingmoe3 dots1 dots3note arcee afmoe laguna ernie4_5 ernie4_5-moe hunyuan-moe hunyuan-dense hunyuan_vl hy_v3 smollm3 gpt-oss lfm2 lfm2moe dream smallthinker llada llada-moe seed_oss grovemoe apertus minimax-01 minimax-m2 minimax-m3 cogvlm rnd1 pangu-embedded mistral3 eagle3 dflash mistral4 paddleocr mimo2 step35 llama-embed maincoder kimi-linear kimi-k3 talkie mellum nanbeige qwen3tts pockettts";

fn validate_model(model: &ModelDesc) -> Result<ValidatedModel<'_>> {
    if model.architecture.len() > MAX_ARCH_BYTES
        || model.architecture.contains('\0')
        || !ARCHITECTURES
            .split(' ')
            .any(|entry| entry == model.architecture)
        || model.layers == 0
        || model.layers > MAX_LAYERS
        || [
            model.embedding,
            model.feed_forward,
            model.heads,
            model.kv_heads,
            model.experts,
            model.head_key,
            model.head_value,
        ]
        .iter()
        .any(|&value| value > MAX_MODEL_PARAM)
    {
        return Err(invalid("invalid quantization model descriptor"));
    }
    Ok(ValidatedModel {
        architecture: &model.architecture,
        layers: usize::try_from(model.layers)?,
        heads: model.heads,
        kv_heads: model.kv_heads,
        experts: model.experts,
    })
}

fn validate_tensor(tensor: &TensorDesc) -> Result<ValidatedTensor> {
    let max_dim = u64::from(i32::MAX.unsigned_abs());
    let max_elements = i64::MAX.unsigned_abs() / 4;
    if tensor.name.len() > MAX_TENSOR_NAME_INPUT
        || tensor.name.contains('\0')
        || tensor
            .dimensions
            .iter()
            .any(|&dim| dim == 0 || dim > max_dim)
    {
        return Err(invalid("invalid quantization tensor descriptor"));
    }
    tensor
        .dimensions
        .iter()
        .try_fold(1_u64, |acc, &dim| acc.checked_mul(dim))
        .filter(|&count| count <= max_elements)
        .ok_or_else(|| invalid("quantization tensor extent exceeds bound"))?;
    let limit = tensor.name.len().min(MAX_TENSOR_NAME_NATIVE);
    let name = String::from_utf8(tensor.name.as_bytes()[..limit].to_vec())?;
    Ok(ValidatedTensor {
        name,
        dimensions: tensor.dimensions,
    })
}

fn validate_actions(tensors: &[ValidatedTensor], actions: &[Action]) -> Result<()> {
    if actions.len() > MAX_ACTIONS {
        return Err(invalid("quantization actions exceed bound"));
    }
    let mut returned = 0_usize;
    for action in actions {
        match action {
            Action::Allows => {
                returned = returned
                    .checked_add(tensors.len())
                    .ok_or_else(|| invalid("quantization response entries exceed bound"))?;
            }
            Action::Types {
                ftype,
                tensor_indices,
            } => {
                if *ftype >= FTYPE_GUESSED || default_type(*ftype) == TYPE_COUNT {
                    return Err(invalid("unsupported quantization ftype"));
                }
                if tensor_indices.len() > tensors.len()
                    || tensor_indices.iter().any(|&index| index >= tensors.len())
                {
                    return Err(invalid("quantization tensor index exceeds extent"));
                }
                returned = returned
                    .checked_add(tensor_indices.len())
                    .ok_or_else(|| invalid("quantization response entries exceed bound"))?;
            }
        }
        if returned > MAX_RETURNED {
            return Err(invalid("quantization response entries exceed bound"));
        }
    }
    Ok(())
}

fn validate_operation(operation: &Operation) -> Result<ValidatedOp<'_>> {
    match operation {
        Operation::Info { ftypes } => {
            if ftypes.len() > MAX_FTYPES || ftypes.iter().any(|&ftype| ftype >= FTYPE_GUESSED) {
                return Err(invalid("invalid quantization ftype inventory request"));
            }
            Ok(ValidatedOp::Info(ftypes))
        }
        Operation::Model {
            model,
            tensors,
            actions,
        } => {
            let model = validate_model(model)?;
            if tensors.is_empty() || tensors.len() > MAX_TENSORS {
                return Err(invalid("quantization tensor count exceeds bound"));
            }
            let tensors = tensors
                .iter()
                .map(validate_tensor)
                .collect::<Result<Vec<_>>>()?;
            validate_actions(&tensors, actions)?;
            Ok(ValidatedOp::Model {
                model,
                tensors,
                actions,
            })
        }
    }
}

fn allows(tensor: &ValidatedTensor) -> bool {
    let name = tensor.name.as_str();
    tensor.dimensions[1..].iter().any(|&dim| dim != 1)
        && name.rfind("weight").unwrap_or(usize::MAX) == name.len().wrapping_sub(6)
        && name != "position_embd.weight"
        && name != "token_types.weight"
        && !EXCLUSIONS.iter().any(|needle| name.contains(needle))
}

fn category(name: &str) -> Category {
    if name == "output.weight" {
        return Category::Output;
    }
    if matches!(name, "token_embd.weight" | "per_layer_token_embd.weight") {
        return Category::TokenEmbedding;
    }
    for (needle, role) in [
        ("attn_qkv.weight", Category::AttentionQkv),
        ("attn_kv_b.weight", Category::AttentionKvB),
        ("attn_v.weight", Category::AttentionV),
        ("attn_k.weight", Category::AttentionK),
        ("attn_q.weight", Category::AttentionQ),
        ("attn_output.weight", Category::AttentionOutput),
        ("ffn_up", Category::FfnUp),
        ("ffn_gate", Category::FfnGate),
        ("ffn_down", Category::FfnDown),
    ] {
        if name.contains(needle) {
            return role;
        }
    }
    Category::Other
}

fn parse_layer(
    model: &ValidatedModel<'_>,
    tensor: &ValidatedTensor,
    counter: usize,
) -> Result<usize> {
    if model.experts <= 1 {
        return Ok(counter);
    }
    let tail = tensor
        .name
        .strip_prefix("blk.")
        .ok_or_else(|| invalid("expert tensor name has no layer prefix"))?
        .trim_start_matches(|ch: char| ch.is_ascii_whitespace());
    let digits = tail
        .bytes()
        .enumerate()
        .take_while(|(index, byte)| {
            byte.is_ascii_digit() || (*index == 0 && matches!(byte, b'+' | b'-'))
        })
        .count();
    let prefix = tail
        .get(..digits)
        .ok_or_else(|| invalid("invalid expert tensor layer prefix"))?;
    let layer = prefix.parse::<i32>()?;
    let layer = usize::try_from(layer)?;
    if layer >= model.layers {
        return Err(invalid("expert tensor layer exceeds model"));
    }
    Ok(layer)
}

fn block_size(kind: u32) -> Result<u64> {
    type_block(kind).ok_or_else(|| invalid("invalid quantization storage type"))
}

fn select_output_or_tied(
    model: &ValidatedModel<'_>,
    ftype: u32,
    default: u32,
    tensor: &ValidatedTensor,
) -> Result<u32> {
    if ftype == 38
        || model.architecture == "falcon"
        || !tensor.dimensions[0].is_multiple_of(block_size(default)?)
    {
        return Ok(8);
    }
    Ok(if low_iq(ftype) || ftype == 23 {
        13
    } else if default == 8 {
        8
    } else {
        14
    })
}

fn select_low_iq(
    model: &ValidatedModel<'_>,
    ftype: u32,
    default: u32,
    role: Category,
    counters: &mut Counters,
) -> u32 {
    if role.is_attention_v() {
        counters.attention += 1;
        if model.gqa() >= 4 || model.experts >= 4 {
            12
        } else if matches!(ftype, 28 | 29) {
            21
        } else {
            10
        }
    } else if role == Category::AttentionK && model.experts == 8 {
        12
    } else if role == Category::FfnDown {
        let kind = if counters.down < model.layers / 8 {
            if matches!(ftype, 28 | 29) { 21 } else { 10 }
        } else {
            default
        };
        counters.down += 1;
        kind
    } else if role == Category::AttentionOutput {
        if model.experts == 8 {
            13
        } else if matches!(ftype, 24 | 31) {
            16
        } else if matches!(ftype, 28 | 29) {
            21
        } else {
            default
        }
    } else {
        default
    }
}

fn select_attention_v(
    model: &ValidatedModel<'_>,
    ftype: u32,
    default: u32,
    counters: &Counters,
) -> u32 {
    let gqa = model.gqa();
    let mut kind = match ftype {
        10 => {
            if gqa >= 4 {
                12
            } else {
                11
            }
        }
        21 | 22 | 26 if gqa >= 4 => 12,
        23 => {
            if gqa >= 4 {
                12
            } else {
                21
            }
        }
        27 => 12,
        12 => {
            if counters.attention < 2 {
                13
            } else {
                12
            }
        }
        13 => 13,
        25 | 30 if gqa >= 4 => 13,
        15 | 17 if more_bits(counters.attention, counters.attention_count) => 14,
        14 if counters.attention < 4 => 13,
        _ => default,
    };
    if model.architecture == "llama"
        && model.layers == 80
        && model.heads != model.kv_heads
        && matches!(kind, 11 | 12)
    {
        kind = 13;
    }
    if model.experts == 8 {
        kind = 8;
    }
    kind
}

fn select_down(model: &ValidatedModel<'_>, ftype: u32, default: u32, layer: usize) -> u32 {
    let layers = model.layers;
    let falcon = model.architecture == "falcon";
    match ftype {
        10 => 11,
        21 if layer < layers / 8 => 12,
        23 => {
            if layer < layers / 8 {
                12
            } else {
                11
            }
        }
        12 => {
            if layer < layers / 16 {
                13
            } else if !falcon || more_bits(layer, layers) {
                12
            } else {
                11
            }
        }
        27 if layer < layers / 8 || (model.experts == 8 && more_bits(layer, layers)) => 12,
        13 => {
            if falcon {
                12
            } else {
                13
            }
        }
        15 if falcon => {
            if layer < layers / 16 {
                14
            } else if more_bits(layer, layers) {
                13
            } else {
                12
            }
        }
        15 | 17 if more_bits(layer, layers) => 14,
        25 | 30 if layer < layers / 8 => 13,
        14 if !falcon && layer < layers / 8 => 13,
        _ => default,
    }
}

fn select_attention_output(model: &ValidatedModel<'_>, ftype: u32, default: u32) -> u32 {
    if model.architecture == "falcon" {
        return if ftype == 13 { 12 } else { default };
    }
    if model.experts == 8 {
        if matches!(ftype, 10 | 22 | 23 | 11 | 12 | 25 | 14 | 15 | 26 | 27 | 30) {
            13
        } else {
            default
        }
    } else {
        match ftype {
            10 => 11,
            23 => 21,
            12 | 27 => 12,
            13 => 13,
            _ => default,
        }
    }
}

fn select_standard(
    model: &ValidatedModel<'_>,
    ftype: u32,
    default: u32,
    tensor: &ValidatedTensor,
    role: Category,
    counters: &mut Counters,
) -> Result<u32> {
    if role == Category::Output || (counters.tied_embeddings && role == Category::TokenEmbedding) {
        return select_output_or_tied(model, ftype, default, tensor);
    }
    if ftype == 38 {
        let mxfp4 = tensor.dimensions[2] > 1
            && (model.architecture != "bailingmoe3"
                || matches!(
                    role,
                    Category::FfnUp | Category::FfnGate | Category::FfnDown
                ));
        return Ok(if mxfp4 { 39 } else { 8 });
    }
    if role == Category::TokenEmbedding {
        return Ok(match ftype {
            19 | 20 | 24 | 31 => 10,
            23 | 28 | 29 => 21,
            36 | 37 | 41 | 141 | 143 => 12,
            _ => default,
        });
    }
    if low_iq(ftype) {
        return Ok(select_low_iq(model, ftype, default, role, counters));
    }
    if role.is_attention_v() {
        let kind = select_attention_v(model, ftype, default, counters);
        counters.attention += 1;
        return Ok(kind);
    }
    Ok(match role {
        Category::AttentionK if model.experts == 8 => 8,
        Category::AttentionQ | Category::AttentionK => match ftype {
            22 => 18,
            23 => 22,
            _ => default,
        },
        Category::FfnDown => {
            let layer = parse_layer(model, tensor, counters.down)?;
            let kind = select_down(model, ftype, default, layer);
            counters.down += 1;
            kind
        }
        Category::AttentionOutput => select_attention_output(model, ftype, default),
        Category::FfnGate | Category::FfnUp => {
            let counter = if role == Category::FfnGate {
                &mut counters.gate
            } else {
                &mut counters.up
            };
            let layer = parse_layer(model, tensor, *counter)?;
            *counter += 1;
            if ftype == 22 && layer >= model.layers / 8 && layer < 7 * model.layers / 8 {
                18
            } else {
                default
            }
        }
        _ => default,
    })
}

fn apply_fallback(columns: u64, kind: u32) -> Result<u32> {
    if columns.is_multiple_of(block_size(kind)?) {
        return Ok(kind);
    }
    let candidate = fallback_candidate(kind)
        .ok_or_else(|| invalid("production policy has no fallback for this type"))?;
    Ok(if columns.is_multiple_of(block_size(candidate)?) {
        candidate
    } else {
        1
    })
}

fn compute_types(
    model: &ValidatedModel<'_>,
    ftype: u32,
    selected: &[&ValidatedTensor],
) -> Result<Vec<u32>> {
    let default = default_type(ftype);
    let mut counters = Counters {
        attention_count: selected
            .iter()
            .filter(|tensor| category(&tensor.name).is_attention_v())
            .count(),
        tied_embeddings: !selected
            .iter()
            .any(|tensor| category(&tensor.name) == Category::Output),
        ..Counters::default()
    };
    selected
        .iter()
        .map(|tensor| {
            if matches!(default, 0 | 1 | 30) {
                return Ok(default);
            }
            let role = category(&tensor.name);
            let chosen = select_standard(model, ftype, default, tensor, role, &mut counters)?;
            apply_fallback(tensor.dimensions[0], chosen)
        })
        .collect()
}

fn execute_info(ftypes: &[u32]) -> Value {
    let types = (0..TYPE_COUNT)
        .map(|kind| json!({"Type": kind, "Name": type_name(kind)}))
        .collect::<Vec<_>>();
    let defaults = ftypes
        .iter()
        .map(|&ftype| default_type(ftype))
        .collect::<Vec<_>>();
    json!({
        "TypeCount": TYPE_COUNT,
        "FtypeGuessed": FTYPE_GUESSED,
        "Types": types,
        "Defaults": defaults,
    })
}

fn execute_model(
    model: &ValidatedModel<'_>,
    tensors: &[ValidatedTensor],
    actions: &[Action],
) -> Result<Value> {
    let mut results = Vec::with_capacity(actions.len());
    for action in actions {
        match action {
            Action::Allows => {
                let allowed = tensors.iter().map(allows).collect::<Vec<_>>();
                let names = tensors
                    .iter()
                    .map(|tensor| tensor.name.as_str())
                    .collect::<Vec<_>>();
                results.push(json!({"Allowed": allowed, "Names": names}));
            }
            Action::Types {
                ftype,
                tensor_indices,
            } => {
                let mut selected = Vec::with_capacity(tensor_indices.len());
                for &index in tensor_indices {
                    let tensor = tensors
                        .get(index)
                        .ok_or_else(|| invalid("quantization tensor index exceeds extent"))?;
                    if !allows(tensor) {
                        return Err(invalid("quantization compute requires quantizable tensors"));
                    }
                    selected.push(tensor);
                }
                let assigned = compute_types(model, *ftype, &selected)?;
                results.push(json!({
                    "DefaultType": default_type(*ftype),
                    "Types": assigned,
                }));
            }
        }
    }
    Ok(json!({"Created": true, "Results": results}))
}

/// Returns `true` when every operation in `value` targets `quantization-info` or
/// `quantization-model`.
pub(super) fn handles(value: &Value) -> bool {
    value["Operations"].as_array().is_some_and(|operations| {
        !operations.is_empty()
            && operations.iter().all(|operation| {
                matches!(
                    operation["Kind"].as_str(),
                    Some("quantization-info" | "quantization-model")
                )
            })
    })
}

/// Validates and evaluates a clean-room `quantization-info` / `quantization-model`
/// conformance request.
///
/// # Errors
/// Rejects schema violations, unknown architectures, unsupported ftypes, invalid
/// tensor shapes or indices, non-quantizable tensors in `types` actions, bad
/// expert layer names, and shapes without a valid fallback type.
pub(super) fn execute(value: &Value) -> Result<Value> {
    if !value["Operations"]
        .as_array()
        .is_some_and(|operations| (1..=MAX_OPERATIONS).contains(&operations.len()))
    {
        return Err(invalid(
            "independent quantization operations must number 1 through 128",
        ));
    }
    if value["Operations"].as_array().is_some_and(|operations| {
        operations.iter().any(|operation| {
            operation["Actions"].as_array().is_some_and(|actions| {
                actions.iter().any(|action| {
                    action["Op"].as_str() == Some("allows")
                        && action.as_object().is_none_or(|object| object.len() != 1)
                })
            })
        })
    }) {
        return Err(invalid("allows action forbids extra fields"));
    }
    let request: Request = serde_json::from_value(value.clone())?;
    if request.schema_version != 1 {
        return Err(invalid(
            "independent quantization requires schema version 1",
        ));
    }
    let validated = request
        .operations
        .iter()
        .map(validate_operation)
        .collect::<Result<Vec<_>>>()?;
    let mut results = Vec::with_capacity(validated.len());
    for operation in validated {
        match operation {
            ValidatedOp::Info(ftypes) => results.push(execute_info(ftypes)),
            ValidatedOp::Model {
                model,
                tensors,
                actions,
            } => results.push(execute_model(&model, &tensors, actions)?),
        }
    }
    Ok(json!({"SchemaVersion": 1, "Results": results}))
}

/// Returns provenance metadata for this independent quantization provider.
pub(super) fn identity() -> Value {
    json!({
        "Kind": "independent-rust-quantization-policy",
        "SupportedSourceIds": [ID],
        "UpstreamCommit": super::UPSTREAM,
        "AssertionSource": {"Path": ID, "Sha256": SOURCE_SHA256},
        "Policy": "clean-room-metadata-tensor-quantization-selection-v1",
        "NativeReferenceReplay": false,
        "ProductionRuntimeCalled": false,
        "ExpectedFixtureLookup": false,
        "CompleteSourceGate": false,
        "NativeRetirementQualified": false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model_json(arch: &str, layers: u32, heads: u32, kv_heads: u32, experts: u32) -> Value {
        json!({
            "Architecture": arch,
            "Embedding": 4096,
            "FeedForward": 14336,
            "Layers": layers,
            "Heads": heads,
            "KvHeads": kv_heads,
            "Experts": experts,
            "HeadKey": 128,
            "HeadValue": 128,
        })
    }

    fn tensor_json(name: &str, dims: [u64; 4]) -> Value {
        json!({"Name": name, "Dimensions": dims})
    }

    fn model_request(model: &Value, tensors: &Value, actions: &Value) -> Value {
        json!({
            "SchemaVersion": 1,
            "Operations": [{
                "Kind": "quantization-model",
                "Model": model,
                "Tensors": tensors,
                "Actions": actions,
            }],
        })
    }

    fn run_single_types(model: &Value, tensors: &[Value], ftype: u32) -> Result<Value> {
        let indices = (0..tensors.len()).collect::<Vec<_>>();
        let request = model_request(
            model,
            &json!(tensors),
            &json!([{"Op": "types", "Ftype": ftype, "TensorIndices": indices}]),
        );
        let response = execute(&request)?;
        Ok(response["Results"][0]["Results"][0].clone())
    }

    #[test]
    fn quantization_info_reports_complete_enum_inventory_and_ftype_defaults() -> Result<()> {
        assert_eq!(ID, "tests/test-quant-type-selection.cpp");
        let meta = identity();
        assert_eq!(meta["SupportedSourceIds"], json!([ID]));
        assert_eq!(meta["AssertionSource"]["Sha256"], SOURCE_SHA256);
        assert_eq!(meta["NativeReferenceReplay"], false);
        assert_eq!(meta["ProductionRuntimeCalled"], false);

        assert!(handles(&json!({
            "SchemaVersion": 1,
            "Operations": [{"Kind": "quantization-info", "Ftypes": []}]
        })));
        assert!(!handles(&json!({
            "SchemaVersion": 1,
            "Operations": [{"Kind": "concurrent-log"}]
        })));

        let empty = execute(&json!({
            "SchemaVersion": 1,
            "Operations": [{"Kind": "quantization-info", "Ftypes": []}]
        }))?;
        assert_eq!(empty["Results"][0]["TypeCount"], 144);
        assert_eq!(empty["Results"][0]["FtypeGuessed"], 1024);
        assert_eq!(empty["Results"][0]["Defaults"], json!([]));

        let ftypes = [0, 1, 2, 3, 4, 7, 15, 32, 38, 39, 40, 41, 141, 143, 1023];
        let info = execute(&json!({
            "SchemaVersion": 1,
            "Operations": [{"Kind": "quantization-info", "Ftypes": ftypes}]
        }))?;
        let entry = &info["Results"][0];
        let types = entry["Types"]
            .as_array()
            .ok_or_else(|| invalid("missing Types array"))?;
        assert_eq!(types.len(), 144);
        assert_eq!(types[0], json!({"Type": 0, "Name": "f32"}));
        assert_eq!(types[4], json!({"Type": 4, "Name": "DEPRECATED"}));
        assert_eq!(
            types[31],
            json!({"Type": 31, "Name": "TYPE_Q4_0_4_4 REMOVED, use Q4_0 with runtime repacking"})
        );
        assert_eq!(types[39], json!({"Type": 39, "Name": "mxfp4"}));
        assert_eq!(types[40], json!({"Type": 40, "Name": "nvfp4"}));
        assert_eq!(types[43], json!({"Type": 43, "Name": null}));
        assert_eq!(types[142], json!({"Type": 142, "Name": "pq2_0"}));
        assert_eq!(types[143], json!({"Type": 143, "Name": "ptq1_0"}));
        assert_eq!(
            entry["Defaults"],
            json!([0, 1, 2, 3, 144, 8, 12, 30, 39, 144, 41, 42, 142, 143, 144])
        );
        Ok(())
    }

    #[test]
    fn quantization_allows_filters_exclusions_truncates_names_and_preserves_5byte_quirk()
    -> Result<()> {
        let long_kept = format!("blk.0.{}.weight", "x".repeat(50));
        let long_dropped = format!("blk.0.{}.weight", "x".repeat(60));
        let tensors = json!([
            tensor_json("token_embd.weight", [4096, 32000, 1, 1]),
            tensor_json("output.weight", [4096, 32000, 1, 1]),
            tensor_json("blk.0.attn_norm.weight", [4096, 1, 1, 1]),
            tensor_json("blk.0.ffn_gate_inp.weight", [4096, 8, 1, 1]),
            tensor_json("blk.0.ssm_conv1d.weight", [4, 4096, 1, 1]),
            tensor_json("blk.0.attn_q.bias", [4096, 4, 1, 1]),
            tensor_json("position_embd.weight", [4096, 512, 1, 1]),
            tensor_json("token_types.weight", [4096, 2, 1, 1]),
            tensor_json("abcde", [32, 32, 1, 1]),
            tensor_json("abcd", [32, 32, 1, 1]),
            tensor_json("abcdef", [32, 32, 1, 1]),
            tensor_json(&long_kept, [4096, 4096, 1, 1]),
            tensor_json(&long_dropped, [4096, 4096, 1, 1]),
            tensor_json("blk.0.ffn_down_shexp.weight", [4096, 4096, 1, 1]),
        ]);
        let response = execute(&model_request(
            &model_json("llama", 4, 32, 8, 0),
            &tensors,
            &json!([{"Op": "allows"}]),
        ))?;
        let allows_result = &response["Results"][0]["Results"][0];
        assert_eq!(
            allows_result["Allowed"],
            json!([
                true, true, false, false, false, false, false, false, true, false, false, true,
                false, true
            ])
        );
        let names = allows_result["Names"]
            .as_array()
            .ok_or_else(|| invalid("missing Names array"))?;
        assert_eq!(names[11], long_kept);
        assert_eq!(
            names[12].as_str().map(str::len),
            Some(MAX_TENSOR_NAME_NATIVE)
        );
        Ok(())
    }

    #[test]
    fn quantization_standard_gqa_llama80_falcon_moe_and_direct_types() -> Result<()> {
        let llama80 = model_json("llama", 80, 64, 8, 0);
        let llama80_tensors = [
            tensor_json("token_embd.weight", [4096, 32000, 1, 1]),
            tensor_json("blk.0.attn_q.weight", [4096, 8192, 1, 1]),
            tensor_json("blk.0.attn_k.weight", [4096, 1024, 1, 1]),
            tensor_json("blk.0.attn_v.weight", [4096, 1024, 1, 1]),
            tensor_json("blk.0.attn_output.weight", [8192, 4096, 1, 1]),
            tensor_json("blk.0.ffn_gate.weight", [4096, 14336, 1, 1]),
            tensor_json("blk.0.ffn_up.weight", [4096, 14336, 1, 1]),
            tensor_json("blk.0.ffn_down.weight", [14336, 4096, 1, 1]),
            tensor_json("output.weight", [4096, 32000, 1, 1]),
        ];
        let q2k = run_single_types(&llama80, &llama80_tensors, 10)?;
        assert_eq!(q2k["DefaultType"], 10);
        assert_eq!(q2k["Types"], json!([10, 10, 10, 13, 11, 10, 10, 11, 14]));
        let bf16 = run_single_types(&llama80, &llama80_tensors, 32)?;
        assert_eq!(bf16["DefaultType"], 30);
        assert_eq!(bf16["Types"], json!(vec![30; 9]));

        let mixtral = model_json("llama", 32, 32, 8, 8);
        let moe_tensors = [
            tensor_json("blk.0.attn_k.weight", [4096, 1024, 1, 1]),
            tensor_json("blk.0.attn_v.weight", [4096, 1024, 1, 1]),
            tensor_json("blk.0.attn_output.weight", [4096, 4096, 1, 1]),
            tensor_json("blk.0.ffn_down_exps.weight", [14336, 4096, 8, 1]),
            tensor_json("blk.5.ffn_down_exps.weight", [14336, 4096, 8, 1]),
            tensor_json("blk.6.ffn_down_exps.weight", [14336, 4096, 8, 1]),
        ];
        let iq3m = run_single_types(&mixtral, &moe_tensors, 27)?;
        assert_eq!(iq3m["Types"], json!([8, 8, 13, 12, 21, 12]));

        let falcon = model_json("falcon", 32, 64, 64, 0);
        let falcon_tensors = [
            tensor_json("output.weight", [4096, 65024, 1, 1]),
            tensor_json("blk.0.attn_output.weight", [4096, 4096, 1, 1]),
            tensor_json("blk.0.ffn_down.weight", [16384, 4096, 1, 1]),
            tensor_json("blk.1.ffn_down.weight", [16384, 4096, 1, 1]),
            tensor_json("blk.2.ffn_down.weight", [16384, 4096, 1, 1]),
            tensor_json("blk.3.ffn_down.weight", [16384, 4096, 1, 1]),
            tensor_json("blk.4.ffn_down.weight", [16384, 4096, 1, 1]),
        ];
        let falcon_q3k_m = run_single_types(&falcon, &falcon_tensors, 12)?;
        assert_eq!(falcon_q3k_m["Types"], json!([8, 11, 13, 13, 12, 12, 11]));
        Ok(())
    }

    #[test]
    fn quantization_tied_embeddings_mxfp4_and_iq2s_layer_counters() -> Result<()> {
        let qwen = model_json("qwen3", 8, 16, 4, 0);
        let tied = run_single_types(
            &qwen,
            &[
                tensor_json("token_embd.weight", [4096, 32000, 1, 1]),
                tensor_json("blk.0.attn_qkv.weight", [4096, 6144, 1, 1]),
                tensor_json("blk.0.attn_kv_b.weight", [4096, 2048, 1, 1]),
                tensor_json("blk.0.ffn_down.weight", [11008, 4096, 1, 1]),
                tensor_json("blk.0.ffn_gate.weight", [4096, 11008, 1, 1]),
                tensor_json("blk.1.ffn_gate.weight", [4096, 11008, 1, 1]),
            ],
            22,
        )?;
        assert_eq!(tied["Types"], json!([14, 12, 12, 21, 21, 18]));

        let bailing = model_json("bailingmoe3", 8, 16, 4, 4);
        let mxfp4 = run_single_types(
            &bailing,
            &[
                tensor_json("blk.0.attn_q.weight", [256, 256, 4, 1]),
                tensor_json("blk.0.ffn_up_exps.weight", [256, 256, 4, 1]),
                tensor_json("blk.0.ffn_down_shexp.weight", [256, 256, 1, 1]),
            ],
            38,
        )?;
        assert_eq!(mxfp4["Types"], json!([8, 39, 8]));

        let empty_actions = execute(&model_request(
            &qwen,
            &json!([tensor_json("blk.0.attn_q.weight", [256, 256, 1, 1])]),
            &json!([]),
        ))?;
        assert_eq!(
            empty_actions["Results"][0],
            json!({"Created": true, "Results": []})
        );
        Ok(())
    }

    #[test]
    fn quantization_low_iq_and_non_multiple_shape_fallbacks() -> Result<()> {
        let iq2m = run_single_types(
            &model_json("llama", 16, 16, 16, 0),
            &[
                tensor_json("token_embd.weight", [256, 1000, 1, 1]),
                tensor_json("output.weight", [256, 1000, 1, 1]),
                tensor_json("blk.0.attn_v.weight", [256, 256, 1, 1]),
                tensor_json("blk.0.attn_output.weight", [256, 256, 1, 1]),
                tensor_json("blk.0.ffn_down.weight", [256, 256, 1, 1]),
                tensor_json("blk.1.ffn_down.weight", [256, 256, 1, 1]),
                tensor_json("blk.2.ffn_down.weight", [256, 256, 1, 1]),
            ],
            28,
        )?;
        assert_eq!(iq2m["Types"], json!([21, 13, 21, 21, 21, 21, 17]));

        let dense = model_json("llama", 8, 8, 8, 0);
        for (ftype, cols, expected) in [(23, 288, 20), (15, 288, 6), (15, 270, 1), (141, 288, 2)] {
            assert_eq!(
                run_single_types(
                    &dense,
                    &[tensor_json("blk.0.attn_k.weight", [cols, 256, 1, 1])],
                    ftype,
                )?["Types"],
                json!([expected])
            );
        }
        assert_eq!(
            run_single_types(
                &dense,
                &[tensor_json("output.weight", [32, 1000, 1, 1])],
                15,
            )?["Types"],
            json!([8])
        );
        Ok(())
    }

    #[test]
    fn quantization_rejects_invalid_envelopes_models_and_tensor_bounds() {
        let model = model_json("llama", 4, 8, 8, 0);
        let good = tensor_json("blk.0.attn_q.weight", [256, 256, 1, 1]);
        let mut utf8_cut = "a".repeat(62);
        utf8_cut.push('\u{00e9}');

        for request in [
            json!({"SchemaVersion": 2, "Operations": [{"Kind": "quantization-info", "Ftypes": []}]}),
            json!({"SchemaVersion": 1, "Operations": []}),
            json!({"SchemaVersion": 1, "Operations": vec![json!({"Kind": "quantization-info", "Ftypes": []}); 129]}),
            json!({"SchemaVersion": 1, "Operations": [{"Kind": "quantization-info", "Ftypes": [1024]}]}),
            json!({"SchemaVersion": 1, "Operations": [{"Kind": "quantization-info", "Ftypes": vec![0; 1025]}]}),
            json!({"SchemaVersion": 1, "Operations": [{"Kind": "quantization-info", "Ftypes": []}], "Extra": 1}),
            model_request(
                &model_json("unknown-arch", 4, 8, 8, 0),
                &json!([&good]),
                &json!([]),
            ),
            model_request(
                &model_json("llama", 0, 8, 8, 0),
                &json!([&good]),
                &json!([]),
            ),
            model_request(
                &model_json("llama", 513, 8, 8, 0),
                &json!([&good]),
                &json!([]),
            ),
            model_request(&model, &json!([]), &json!([])),
            model_request(
                &model,
                &json!([tensor_json("blk.0.attn_q.weight", [0, 256, 1, 1])]),
                &json!([]),
            ),
            model_request(
                &model,
                &json!([tensor_json(
                    "blk.0.attn_q.weight",
                    [2_000_000_000, 2_000_000_000, 1, 1]
                )]),
                &json!([]),
            ),
            model_request(
                &model,
                &json!([tensor_json(&utf8_cut, [256, 256, 1, 1])]),
                &json!([]),
            ),
        ] {
            assert!(execute(&request).is_err());
        }
    }

    #[test]
    fn quantization_rejects_invalid_actions_unquantizable_targets_and_bad_shapes() {
        let model = model_json("llama", 4, 8, 8, 0);
        let good = tensor_json("blk.0.attn_q.weight", [256, 256, 1, 1]);
        let norm = tensor_json("blk.0.attn_norm.weight", [256, 4, 1, 1]);

        for request in [
            model_request(
                &model,
                &json!([&good]),
                &json!([{"Op": "allows", "Extra": 1}]),
            ),
            model_request(&model, &json!([&good]), &json!([{"Op": "unknown"}])),
            model_request(
                &model,
                &json!([&good]),
                &json!([{"Op": "types", "Ftype": 39, "TensorIndices": [0]}]),
            ),
            model_request(
                &model,
                &json!([&good]),
                &json!([{"Op": "types", "Ftype": 4, "TensorIndices": [0]}]),
            ),
            model_request(
                &model,
                &json!([&good]),
                &json!([{"Op": "types", "Ftype": 15, "TensorIndices": [1]}]),
            ),
            model_request(
                &model,
                &json!([&good]),
                &json!(vec![json!({"Op": "allows"}); 129]),
            ),
            model_request(
                &model,
                &json!([norm]),
                &json!([{"Op": "types", "Ftype": 15, "TensorIndices": [0]}]),
            ),
            model_request(
                &model,
                &json!([tensor_json("blk.0.attn_q.weight", [33, 256, 1, 1])]),
                &json!([{"Op": "types", "Ftype": 7, "TensorIndices": [0]}]),
            ),
            model_request(
                &model,
                &json!([tensor_json("blk.0.attn_q.weight", [64, 256, 1, 1])]),
                &json!([{"Op": "types", "Ftype": 40, "TensorIndices": [0]}]),
            ),
            model_request(
                &model_json("qwen3moe", 4, 8, 4, 8),
                &json!([tensor_json("blk.9.ffn_down_exps.weight", [256, 256, 8, 1])]),
                &json!([{"Op": "types", "Ftype": 15, "TensorIndices": [0]}]),
            ),
            model_request(
                &model_json("qwen3moe", 4, 8, 4, 8),
                &json!([tensor_json("bad.ffn_down_exps.weight", [256, 256, 8, 1])]),
                &json!([{"Op": "types", "Ftype": 15, "TensorIndices": [0]}]),
            ),
        ] {
            assert!(execute(&request).is_err());
        }
    }
}
