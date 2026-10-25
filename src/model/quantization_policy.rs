//! Metadata-only tensor quantization policy from pinned `llama-quant.cpp`.
//!
//! No weights, model snapshots, expected schemas or codec execution are needed.
//! Tensor categories, tied embeddings, per-layer counters, architecture-sensitive
//! precision mixtures and shape fallbacks follow the production policy.

use super::gguf_context::tensor_layout;
use crate::{Result, invalid};

/// Pinned public GGML type enum extent, including removed and reserved entries.
pub const TYPE_COUNT: u32 = 144;

/// Native type name, retaining removed entries and reserved gaps.
#[must_use]
pub const fn type_name(kind: u32) -> Option<&'static str> {
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

/// Native default GGML type for an ftype; unsupported ftypes return `TYPE_COUNT`.
#[must_use]
pub const fn default_type(ftype: u32) -> u32 {
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

/// Architecture and hyperparameters sufficient for native precision selection.
#[derive(Clone, Debug)]
pub struct ModelDescriptor {
    /// Pinned architecture identifier.
    pub architecture: String,
    /// Model embedding width.
    pub embedding: u32,
    /// Feed-forward width.
    pub feed_forward: u32,
    /// Total layer count.
    pub layers: u32,
    /// Attention heads per layer.
    pub heads: u32,
    /// Shared KV attention heads per layer.
    pub kv_heads: u32,
    /// Number of experts; zero denotes a dense model.
    pub experts: u32,
    /// Key head width.
    pub head_key: u32,
    /// Value head width.
    pub head_value: u32,
}

/// Independently owned input tensor descriptor, without allocated weights.
#[derive(Clone, Debug)]
pub struct TensorDescriptor {
    name: String,
    dimensions: [u64; 4],
}

impl TensorDescriptor {
    /// Creates a checked native F32 tensor descriptor, truncating names to 63 bytes.
    ///
    /// # Errors
    /// Rejects NUL/oversized names, truncated UTF-8 and nonpositive or overflowing shapes.
    pub fn new(name: &str, dimensions: [u64; 4]) -> Result<Self> {
        if name.len() > 1024
            || name.contains('\0')
            || dimensions
                .iter()
                .any(|&value| value == 0 || value > i32::MAX.unsigned_abs().into())
        {
            return Err(invalid("invalid quantization tensor descriptor"));
        }
        dimensions
            .iter()
            .try_fold(1_u64, |count, &value| count.checked_mul(value))
            .filter(|&count| count <= i64::MAX.unsigned_abs() / 4)
            .ok_or_else(|| invalid("quantization tensor extent exceeds capacity"))?;
        Ok(Self {
            name: String::from_utf8(name.as_bytes()[..name.len().min(63)].to_vec())?,
            dimensions,
        })
    }
    /// Effective native tensor name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }
    /// Four contiguous tensor dimensions.
    #[must_use]
    pub const fn dimensions(&self) -> [u64; 4] {
        self.dimensions
    }
}

/// Explicit native policy parameters; defaults match the production quantizer.
#[derive(Clone, Debug)]
#[allow(
    clippy::struct_excessive_bools,
    reason = "These are independent native quantization flags, not mutually exclusive state transitions."
)]
pub struct Parameters {
    /// Copy tensors without quantization.
    pub only_copy: bool,
    /// Admit the output matrix to quantization.
    pub quantize_output_tensor: bool,
    /// Suppress automatic precision mixtures.
    pub pure: bool,
    /// Explicit token embedding storage type.
    pub token_embedding_type: Option<u32>,
    /// Explicit output storage type.
    pub output_tensor_type: Option<u32>,
    /// Whether importance weights are available for precision decisions.
    pub has_importance: bool,
}
impl Default for Parameters {
    fn default() -> Self {
        Self {
            only_copy: false,
            quantize_output_tensor: true,
            pure: false,
            token_embedding_type: None,
            output_tensor_type: None,
            has_importance: false,
        }
    }
}

/// Reusable metadata-based quantization policy; computations reset all counters.
#[derive(Clone, Debug)]
pub struct Policy {
    model: ModelDescriptor,
    parameters: Parameters,
    layers: usize,
}
impl Policy {
    /// Validates a model and explicit quantization policy parameters.
    ///
    /// # Errors
    /// Rejects unknown architectures, invalid layer counts, excessive hyperparameters
    /// and removed/unknown override storage types.
    pub fn new(model: ModelDescriptor, parameters: Parameters) -> Result<Self> {
        if !ARCHITECTURES
            .split_whitespace()
            .any(|name| name == model.architecture)
            || model.layers == 0
            || model.layers > 512
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
            .any(|&value| value > 1_048_576)
            || [
                parameters.token_embedding_type,
                parameters.output_tensor_type,
            ]
            .into_iter()
            .flatten()
            .any(|kind| tensor_layout(kind).is_none())
        {
            return Err(invalid("invalid quantization model or parameters"));
        }
        let layers = usize::try_from(model.layers)?;
        Ok(Self {
            model,
            parameters,
            layers,
        })
    }

    /// Whether this descriptor satisfies the native quantization eligibility policy.
    #[must_use]
    pub fn allows(&self, tensor: &TensorDescriptor) -> bool {
        let name = tensor.name();
        !self.parameters.only_copy
            && tensor.dimensions[1..]
                .iter()
                .any(|&dimension| dimension != 1)
            && name.rfind("weight").unwrap_or(usize::MAX) == name.len().wrapping_sub(6)
            && (self.parameters.quantize_output_tensor || name != "output.weight")
            && name != "position_embd.weight"
            && name != "token_types.weight"
            && !EXCLUSIONS.iter().any(|needle| name.contains(needle))
    }

    /// Assigns native target types in supplied tensor order, with fresh counters.
    ///
    /// # Errors
    /// Rejects unsupported ftypes, excessive descriptor counts, malformed expert
    /// layer names and shapes with no production fallback type.
    pub fn compute(&self, ftype: u32, tensors: &[&TensorDescriptor]) -> Result<Vec<u32>> {
        let default = default_type(ftype);
        if default == TYPE_COUNT || tensors.len() > 32768 {
            return Err(invalid("unsupported quantization ftype or tensor extent"));
        }
        let mut counters = Counters {
            attention_count: tensors
                .iter()
                .filter(|tensor| category(tensor.name()).attention_v())
                .count(),
            tied_embeddings: !tensors
                .iter()
                .any(|tensor| category(tensor.name()) == Category::Output),
            ..Counters::default()
        };
        tensors
            .iter()
            .map(|tensor| {
                if !self.allows(tensor) {
                    return Ok(0);
                }
                let role = category(tensor.name());
                if role == Category::TokenEmbedding
                    && let Some(kind) = self.parameters.token_embedding_type
                {
                    return Ok(kind);
                }
                if role == Category::Output
                    && let Some(kind) = self.parameters.output_tensor_type
                {
                    return Ok(kind);
                }
                if matches!(default, 0 | 1 | 30) {
                    return Ok(default);
                }
                let chosen = if self.parameters.pure {
                    default
                } else {
                    self.standard(ftype, default, tensor, role, &mut counters)?
                };
                fallback(tensor.dimensions[0], chosen)
            })
            .collect()
    }

    fn gqa(&self) -> u32 {
        self.model
            .heads
            .checked_div(self.model.kv_heads)
            .unwrap_or(0)
    }
    fn layer(&self, tensor: &TensorDescriptor, counter: usize) -> Result<usize> {
        if self.model.experts <= 1 {
            return Ok(counter);
        }
        let value = tensor
            .name()
            .strip_prefix("blk.")
            .ok_or_else(|| invalid("expert tensor name has no layer prefix"))?
            .trim_start_matches(|character: char| character.is_ascii_whitespace());
        let count = value
            .bytes()
            .enumerate()
            .take_while(|(index, byte)| {
                byte.is_ascii_digit() || (*index == 0 && matches!(byte, b'+' | b'-'))
            })
            .count();
        let layer = value[..count].parse::<i32>()?;
        let layer = usize::try_from(layer)?;
        if layer >= usize::try_from(self.model.layers)? {
            return Err(invalid("expert tensor layer exceeds model"));
        }
        Ok(layer)
    }
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
    const fn attention_v(self) -> bool {
        matches!(
            self,
            Self::AttentionV | Self::AttentionQkv | Self::AttentionKvB
        )
    }
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

#[derive(Default)]
struct Counters {
    attention_count: usize,
    attention: usize,
    down: usize,
    gate: usize,
    up: usize,
    tied_embeddings: bool,
}
const fn more_bits(layer: usize, layers: usize) -> bool {
    layer < layers / 8 || layer >= 7 * layers / 8 || (layer - layers / 8) % 3 == 2
}
const fn low_iq(ftype: u32) -> bool {
    matches!(ftype, 19 | 20 | 24 | 28 | 29 | 31)
}

impl Policy {
    fn standard(
        &self,
        ftype: u32,
        default: u32,
        tensor: &TensorDescriptor,
        role: Category,
        counters: &mut Counters,
    ) -> Result<u32> {
        if role == Category::Output
            || (counters.tied_embeddings && role == Category::TokenEmbedding)
        {
            if let Some(kind) = self.parameters.output_tensor_type {
                return Ok(kind);
            }
            return Ok(
                if ftype == 38
                    || self.model.architecture == "falcon"
                    || !tensor.dimensions[0].is_multiple_of(block(default)?)
                {
                    8
                } else if low_iq(ftype) || ftype == 23 {
                    13
                } else if default == 8 {
                    8
                } else {
                    14
                },
            );
        }
        if ftype == 38 {
            return Ok(
                if tensor.dimensions[2] > 1
                    && (self.model.architecture != "bailingmoe3"
                        || matches!(
                            role,
                            Category::FfnUp | Category::FfnGate | Category::FfnDown
                        ))
                {
                    39
                } else {
                    8
                },
            );
        }
        if role == Category::TokenEmbedding {
            return Ok(match ftype {
                19 | 20 | 24 | 31 => 10,
                28 | 29 | 23 => 21,
                36 | 37 | 41 | 141 | 143 => 12,
                _ => default,
            });
        }
        if low_iq(ftype) {
            return Ok(self.low_iq_type(ftype, default, role, counters));
        }
        if role.attention_v() {
            let kind = self.attention_v_type(ftype, default, counters);
            counters.attention += 1;
            return Ok(kind);
        }
        Ok(match role {
            Category::AttentionK if self.model.experts == 8 => 8,
            Category::AttentionQ | Category::AttentionK => match ftype {
                22 => 18,
                23 => 22,
                _ => default,
            },
            Category::FfnDown => {
                let layer = self.layer(tensor, counters.down)?;
                let kind = self.down_type(ftype, default, layer);
                counters.down += 1;
                kind
            }
            Category::AttentionOutput => self.attention_output_type(ftype, default),
            Category::FfnGate | Category::FfnUp => {
                let counter = if role == Category::FfnGate {
                    &mut counters.gate
                } else {
                    &mut counters.up
                };
                let layer = self.layer(tensor, *counter)?;
                *counter += 1;
                if ftype == 22 && layer >= self.layers / 8 && layer < 7 * self.layers / 8 {
                    18
                } else {
                    default
                }
            }
            _ => default,
        })
    }

    fn low_iq_type(
        &self,
        ftype: u32,
        default: u32,
        role: Category,
        counters: &mut Counters,
    ) -> u32 {
        if role.attention_v() {
            counters.attention += 1;
            if self.gqa() >= 4 || self.model.experts >= 4 {
                12
            } else if matches!(ftype, 28 | 29) {
                21
            } else {
                10
            }
        } else if role == Category::AttentionK && self.model.experts == 8 {
            12
        } else if role == Category::FfnDown {
            let kind = if counters.down < self.layers / 8 {
                if matches!(ftype, 28 | 29) { 21 } else { 10 }
            } else {
                default
            };
            counters.down += 1;
            kind
        } else if role == Category::AttentionOutput {
            if self.model.experts == 8 {
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

    fn attention_v_type(&self, ftype: u32, default: u32, counters: &Counters) -> u32 {
        let mut kind = match ftype {
            10 => {
                if self.gqa() >= 4 {
                    12
                } else {
                    11
                }
            }
            21 | 22 | 26 if self.gqa() >= 4 => 12,
            23 => {
                if self.gqa() >= 4 {
                    12
                } else if self.parameters.has_importance {
                    18
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
            25 | 30 if self.gqa() >= 4 => 13,
            15 | 17 if more_bits(counters.attention, counters.attention_count) => 14,
            14 if counters.attention < 4 => 13,
            _ => default,
        };
        if self.model.architecture == "llama"
            && self.model.layers == 80
            && self.model.heads != self.model.kv_heads
            && matches!(kind, 11 | 12)
        {
            kind = 13;
        }
        if self.model.experts == 8 {
            kind = 8;
        }
        kind
    }

    fn down_type(&self, ftype: u32, default: u32, layer: usize) -> u32 {
        let layers = self.layers;
        let falcon = self.model.architecture == "falcon";
        match ftype {
            10 => 11,
            21 if layer < layers / 8 => 12,
            23 if !self.parameters.has_importance => {
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
            27 if layer < layers / 8 || (self.model.experts == 8 && more_bits(layer, layers)) => 12,
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
            25 | 30 if layer < layers / 8 && !self.parameters.has_importance => 13,
            14 if !falcon && layer < layers / 8 => 13,
            2 | 8 if self.parameters.has_importance && layer < layers / 8 => {
                if ftype == 2 {
                    3
                } else {
                    7
                }
            }
            _ => default,
        }
    }

    fn attention_output_type(&self, ftype: u32, default: u32) -> u32 {
        if self.model.architecture == "falcon" {
            return if ftype == 13 { 12 } else { default };
        }
        if self.model.experts == 8 {
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
}

fn block(kind: u32) -> Result<u64> {
    tensor_layout(kind)
        .map(|(block, _)| block)
        .ok_or_else(|| invalid("invalid quantization storage type"))
}
fn fallback(columns: u64, kind: u32) -> Result<u32> {
    if columns.is_multiple_of(block(kind)?) {
        return Ok(kind);
    }
    let kind = match kind {
        19 | 29 | 16 | 17 | 22 | 18 | 21 | 23 => 20,
        42 | 143 | 142 | 10 | 11 | 34 | 35 => 2,
        12 => 6,
        13 => 7,
        14 => 8,
        _ => return Err(invalid("production policy has no fallback for this type")),
    };
    Ok(if columns.is_multiple_of(block(kind)?) {
        kind
    } else {
        1
    })
}

// Names are copied from the pinned production LLM_ARCH_NAMES table.
const ARCHITECTURES: &str = "clip llama llama4 deci falcon grok gpt2 gptj gptneox mpt baichuan starcoder refact bert modern-bert nomic-bert nomic-bert-moe neo-bert jina-bert-v2 jina-bert-v3 eurobert bloom stablelm qwen qwen2 qwen2moe qwen2vl qwen3 qwen3moe qwen3next qwen3vl qwen3vlmoe qwen35 qwen35moe dspark phi2 phi3 phimoe plamo plamo2 plamo3 codeshell orion internlm2 minicpm minicpm3 gemma gemma2 gemma3 gemma3n gemma4 gemma4-assistant gemma-embedding starcoder2 mamba mamba2 jamba falcon-h1 xverse command-r cohere2 cohere2moe dbrx olmo olmo2 olmoe muse-glimmer openelm arctic deepseek deepseek2 deepseek2-ocr deepseek32 deepseek4 chatglm glm4 glm4moe glm-dsa bitnet t5 t5encoder jais jais2 nemotron nemotron_h nemotron_h_moe exaone exaone4 exaone-moe rwkv6 rwkv6qwen2 rwkv7 arwkv7 granite granitemoe granitehybrid graniteswitch granite_swa chameleon wavtokenizer-dec plm bailingmoe bailingmoe2 bailingmoe3 dots1 dots3note arcee afmoe laguna ernie4_5 ernie4_5-moe hunyuan-moe hunyuan-dense hunyuan_vl hy_v3 smollm3 gpt-oss lfm2 lfm2moe dream smallthinker llada llada-moe seed_oss grovemoe apertus minimax-01 minimax-m2 minimax-m3 cogvlm rnd1 pangu-embedded mistral3 eagle3 dflash mistral4 paddleocr mimo2 step35 llama-embed maincoder kimi-linear kimi-k3 talkie mellum nanbeige qwen3tts pockettts";
