//! Clean-room GGML tensor metadata, type descriptors, and IEEE-754 / quantized scalar codecs.
//!
//! No `bonsai_prism_runtime`, `bonsai_gguf`, or native GGML libraries are linked or called.

use serde_json::{Value, json};

use crate::{Result, invalid};

pub(super) const GGML_MAX_DIMS: usize = 4;
pub(super) const GGML_MAX_SRC: usize = 10;
pub(super) const GGML_MAX_OP_PARAMS: usize = 64;

pub(super) const TYPE_F32: i32 = 0;
pub(super) const TYPE_F16: i32 = 1;
pub(super) const TYPE_Q4_0: i32 = 2;
pub(super) const TYPE_Q4_1: i32 = 3;
pub(super) const TYPE_Q5_0: i32 = 6;
pub(super) const TYPE_Q5_1: i32 = 7;
pub(super) const TYPE_Q8_0: i32 = 8;
pub(super) const TYPE_Q8_1: i32 = 9;
pub(super) const TYPE_Q2_K: i32 = 10;
pub(super) const TYPE_Q3_K: i32 = 11;
pub(super) const TYPE_Q4_K: i32 = 12;
pub(super) const TYPE_Q5_K: i32 = 13;
pub(super) const TYPE_Q6_K: i32 = 14;
pub(super) const TYPE_Q8_K: i32 = 15;
pub(super) const TYPE_IQ2_XXS: i32 = 16;
pub(super) const TYPE_IQ2_XS: i32 = 17;
pub(super) const TYPE_IQ1_S: i32 = 19;
pub(super) const TYPE_IQ4_NL: i32 = 20;
pub(super) const TYPE_IQ2_S: i32 = 22;
pub(super) const TYPE_IQ4_XS: i32 = 23;
pub(super) const TYPE_I8: i32 = 24;
pub(super) const TYPE_I16: i32 = 25;
pub(super) const TYPE_I32: i32 = 26;
pub(super) const TYPE_I64: i32 = 27;
pub(super) const TYPE_F64: i32 = 28;
pub(super) const TYPE_IQ1_M: i32 = 29;
pub(super) const TYPE_BF16: i32 = 30;
pub(super) const TYPE_TQ1_0: i32 = 34;
pub(super) const TYPE_TQ2_0: i32 = 35;
pub(super) const TYPE_MXFP4: i32 = 39;
pub(super) const TYPE_Q1_0: i32 = 41;
pub(super) const TYPE_Q2_0: i32 = 42;

pub(super) const OP_NONE: i32 = 0;
pub(super) const OP_DUP: i32 = 1;
pub(super) const OP_ADD: i32 = 2;
pub(super) const OP_ADD_ID: i32 = 3;
pub(super) const OP_ACC: i32 = 5;
pub(super) const OP_SUB: i32 = 6;
pub(super) const OP_MUL: i32 = 7;
pub(super) const OP_DIV: i32 = 8;
pub(super) const OP_SQR: i32 = 9;
pub(super) const OP_SQRT: i32 = 10;
pub(super) const OP_LOG: i32 = 11;
pub(super) const OP_SIN: i32 = 12;
pub(super) const OP_COS: i32 = 13;
pub(super) const OP_SUM: i32 = 14;
pub(super) const OP_SUM_ROWS: i32 = 15;
pub(super) const OP_CUMSUM: i32 = 16;
pub(super) const OP_MEAN: i32 = 17;
pub(super) const OP_ARGMAX: i32 = 18;
pub(super) const OP_COUNT_EQUAL: i32 = 19;
pub(super) const OP_REPEAT: i32 = 20;
pub(super) const OP_REPEAT_BACK: i32 = 21;
pub(super) const OP_CONCAT: i32 = 22;
pub(super) const OP_SILU_BACK: i32 = 23;
pub(super) const OP_NORM: i32 = 24;
pub(super) const OP_RMS_NORM: i32 = 25;
pub(super) const OP_RMS_NORM_BACK: i32 = 26;
pub(super) const OP_GROUP_NORM: i32 = 27;
pub(super) const OP_L2_NORM: i32 = 28;
pub(super) const OP_MUL_MAT: i32 = 29;
pub(super) const OP_MUL_MAT_ID: i32 = 30;
pub(super) const OP_OUT_PROD: i32 = 31;
pub(super) const OP_SCALE: i32 = 32;
pub(super) const OP_SET: i32 = 33;
pub(super) const OP_CPY: i32 = 34;
pub(super) const OP_CONT: i32 = 35;
pub(super) const OP_RESHAPE: i32 = 36;
pub(super) const OP_VIEW: i32 = 37;
pub(super) const OP_PERMUTE: i32 = 38;
pub(super) const OP_TRANSPOSE: i32 = 39;
pub(super) const OP_GET_ROWS: i32 = 40;
pub(super) const OP_GET_ROWS_BACK: i32 = 41;
pub(super) const OP_SET_ROWS: i32 = 42;
pub(super) const OP_DIAG: i32 = 43;
pub(super) const OP_DIAG_MASK_INF: i32 = 44;
pub(super) const OP_SOFT_MAX: i32 = 46;
pub(super) const OP_SOFT_MAX_BACK: i32 = 47;
pub(super) const OP_ROPE: i32 = 48;
pub(super) const OP_ROPE_BACK: i32 = 49;
pub(super) const OP_CLAMP: i32 = 50;
pub(super) const OP_CONV_TRANSPOSE_1D: i32 = 51;
pub(super) const OP_IM2COL: i32 = 52;
pub(super) const OP_IM2COL_3D: i32 = 54;
pub(super) const OP_COL2IM_1D: i32 = 55;
pub(super) const OP_CONV_2D: i32 = 56;
pub(super) const OP_CONV_3D: i32 = 57;
pub(super) const OP_CONV_2D_DW: i32 = 58;
pub(super) const OP_CONV_TRANSPOSE_2D: i32 = 59;
pub(super) const OP_POOL_1D: i32 = 60;
pub(super) const OP_POOL_2D: i32 = 61;
pub(super) const OP_UPSCALE: i32 = 63;
pub(super) const OP_PAD: i32 = 64;
pub(super) const OP_PAD_REFLECT_1D: i32 = 65;
pub(super) const OP_ROLL: i32 = 66;
pub(super) const OP_ARANGE: i32 = 67;
pub(super) const OP_TIMESTEP_EMBEDDING: i32 = 68;
pub(super) const OP_ARGSORT: i32 = 69;
pub(super) const OP_TOP_K: i32 = 70;
pub(super) const OP_LEAKY_RELU: i32 = 71;
pub(super) const OP_TRI: i32 = 72;
pub(super) const OP_FILL: i32 = 73;
pub(super) const OP_FLASH_ATTN_EXT: i32 = 74;
pub(super) const OP_SSM_CONV: i32 = 76;
pub(super) const OP_SSM_SCAN: i32 = 77;
pub(super) const OP_RWKV_WKV6: i32 = 82;
pub(super) const OP_GATED_LINEAR_ATTN: i32 = 83;
pub(super) const OP_RWKV_WKV7: i32 = 84;
pub(super) const OP_SOLVE_TRI: i32 = 85;
pub(super) const OP_GATED_DELTA_NET: i32 = 86;
pub(super) const OP_LIGHTNING_INDEXER: i32 = 87;
pub(super) const OP_DSV4_HC_COMB: i32 = 88;
pub(super) const OP_DSV4_HC_PRE: i32 = 89;
pub(super) const OP_DSV4_HC_POST: i32 = 90;
pub(super) const OP_UNARY: i32 = 91;
pub(super) const OP_CROSS_ENTROPY_LOSS: i32 = 96;
pub(super) const OP_CROSS_ENTROPY_LOSS_BACK: i32 = 97;
pub(super) const OP_OPT_STEP_ADAMW: i32 = 98;
pub(super) const OP_OPT_STEP_SGD: i32 = 99;
pub(super) const OP_GLU: i32 = 100;
pub(super) const OP_GATED_DELTA_NET_REPLAY: i32 = 101;

pub(super) const FLAG_PARAM: i32 = 4;
pub(super) const FLAG_LOSS: i32 = 8;

pub(super) const STATUS_SUCCESS: i32 = 0;
pub(super) const SORT_DESC: i32 = 1;
pub(super) const TRI_UPPER: i32 = 1;
pub(super) const TRI_LOWER_DIAG: i32 = 2;
pub(super) const TRI_LOWER: i32 = 3;
pub(super) const SCALE_MODE_BILINEAR: i32 = 1;
pub(super) const SCALE_FLAG_ALIGN_CORNERS: i32 = 256;
pub(super) const ROPE_TYPE_NEOX: i32 = 2;
pub(super) const ROPE_TYPE_MROPE: i32 = 8;
pub(super) const ROPE_TYPE_VISION: i32 = 24;
pub(super) const ROPE_TYPE_IMROPE: i32 = 40;
pub(super) const UNARY_OP_ABS: i32 = 0;
pub(super) const UNARY_OP_SGN: i32 = 1;
pub(super) const UNARY_OP_NEG: i32 = 2;
pub(super) const UNARY_OP_STEP: i32 = 3;
pub(super) const UNARY_OP_TANH: i32 = 4;
pub(super) const UNARY_OP_ELU: i32 = 5;
pub(super) const UNARY_OP_RELU: i32 = 6;
pub(super) const UNARY_OP_SIGMOID: i32 = 7;
pub(super) const UNARY_OP_GELU: i32 = 8;
pub(super) const UNARY_OP_GELU_QUICK: i32 = 9;
pub(super) const UNARY_OP_SILU: i32 = 10;
pub(super) const UNARY_OP_HARDSWISH: i32 = 11;
pub(super) const UNARY_OP_HARDSIGMOID: i32 = 12;
pub(super) const UNARY_OP_EXP: i32 = 13;
pub(super) const UNARY_OP_EXPM1: i32 = 14;
pub(super) const UNARY_OP_SOFTPLUS: i32 = 15;
pub(super) const UNARY_OP_GELU_ERF: i32 = 16;
pub(super) const UNARY_OP_XIELU: i32 = 17;
pub(super) const UNARY_OP_FLOOR: i32 = 18;
pub(super) const UNARY_OP_CEIL: i32 = 19;
pub(super) const UNARY_OP_ROUND: i32 = 20;
pub(super) const UNARY_OP_TRUNC: i32 = 21;

pub(super) const fn blck_size(kind: i32) -> i64 {
    match kind {
        2..=9 | 20 | 39 | 41 | 42 => 32,
        40 | 142 => 64,
        10..=19 | 21..=23 | 29 | 34 | 35 | 143 => 256,
        _ => 1,
    }
}

pub(super) const fn type_size(kind: i32) -> u64 {
    match kind {
        0 | 26 => 4,      // F32 / I32
        1 | 25 | 30 => 2, // F16 / I16 / BF16
        2 | 20 => 18,     // Q4_0 / IQ4_NL
        3 | 142 => 20,    // Q4_1 / PQ2_0
        6 => 22,          // Q5_0
        7 => 24,          // Q5_1
        8 => 34,          // Q8_0
        9 | 40 => 36,     // Q8_1 / NVFP4
        10 => 84,         // Q2_K
        11 | 21 => 110,   // Q3_K / IQ3_S
        12 => 144,        // Q4_K
        13 => 176,        // Q5_K
        14 => 210,        // Q6_K
        15 => 292,        // Q8_K
        16 | 35 => 66,    // IQ2_XXS / TQ2_0
        17 => 74,         // IQ2_XS
        18 => 98,         // IQ3_XXS
        19 => 50,         // IQ1_S
        22 => 82,         // IQ2_S
        23 => 136,        // IQ4_XS
        27 | 28 => 8,     // I64 / F64
        29 | 143 => 56,   // IQ1_M / PTQ1_0
        34 => 54,         // TQ1_0
        39 => 17,         // MXFP4
        41 => 6,          // Q1_0
        42 => 10,         // Q2_0
        _ => 1,
    }
}

pub(super) const fn is_quantized(kind: i32) -> bool {
    blck_size(kind) > 1
}

pub(super) fn type_name(kind: i32) -> &'static str {
    match kind {
        0 => "f32",
        1 => "f16",
        2 => "q4_0",
        3 => "q4_1",
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
        34 => "tq1_0",
        35 => "tq2_0",
        39 => "mxfp4",
        40 => "nvfp4",
        41 => "q1_0",
        42 => "q2_0",
        142 => "pq2_0",
        143 => "ptq1_0",
        _ => "NONE",
    }
}

fn op_name_low(op: i32) -> Option<&'static str> {
    Some(match op {
        0 => "NONE",
        1 => "DUP",
        2 => "ADD",
        3 => "ADD_ID",
        4 => "ADD1",
        5 => "ACC",
        6 => "SUB",
        7 => "MUL",
        8 => "DIV",
        9 => "SQR",
        10 => "SQRT",
        11 => "LOG",
        12 => "SIN",
        13 => "COS",
        14 => "SUM",
        15 => "SUM_ROWS",
        16 => "CUMSUM",
        17 => "MEAN",
        18 => "ARGMAX",
        19 => "COUNT_EQUAL",
        20 => "REPEAT",
        21 => "REPEAT_BACK",
        22 => "CONCAT",
        23 => "SILU_BACK",
        24 => "NORM",
        25 => "RMS_NORM",
        26 => "RMS_NORM_BACK",
        27 => "GROUP_NORM",
        28 => "L2_NORM",
        29 => "MUL_MAT",
        30 => "MUL_MAT_ID",
        31 => "OUT_PROD",
        32 => "SCALE",
        33 => "SET",
        34 => "CPY",
        35 => "CONT",
        36 => "RESHAPE",
        37 => "VIEW",
        38 => "PERMUTE",
        39 => "TRANSPOSE",
        40 => "GET_ROWS",
        41 => "GET_ROWS_BACK",
        42 => "SET_ROWS",
        43 => "DIAG",
        44 => "DIAG_MASK_INF",
        45 => "DIAG_MASK_ZERO",
        46 => "SOFT_MAX",
        47 => "SOFT_MAX_BACK",
        48 => "ROPE",
        49 => "ROPE_BACK",
        _ => return None,
    })
}

pub(super) fn op_name(op: i32) -> &'static str {
    if let Some(name) = op_name_low(op) {
        return name;
    }
    match op {
        50 => "CLAMP",
        51 => "CONV_TRANSPOSE_1D",
        52 => "IM2COL",
        53 => "IM2COL_BACK",
        54 => "IM2COL_3D",
        55 => "COL2IM_1D",
        56 => "CONV_2D",
        57 => "CONV_3D",
        58 => "CONV_2D_DW",
        59 => "CONV_TRANSPOSE_2D",
        60 => "POOL_1D",
        61 => "POOL_2D",
        62 => "POOL_2D_BACK",
        63 => "UPSCALE",
        64 => "PAD",
        65 => "PAD_REFLECT_1D",
        66 => "ROLL",
        67 => "ARANGE",
        68 => "TIMESTEP_EMBEDDING",
        69 => "ARGSORT",
        70 => "TOP_K",
        71 => "LEAKY_RELU",
        72 => "TRI",
        73 => "FILL",
        74 => "FLASH_ATTN_EXT",
        75 => "FLASH_ATTN_BACK",
        76 => "SSM_CONV",
        77 => "SSM_SCAN",
        78 => "WIN_PART",
        79 => "WIN_UNPART",
        80 => "GET_REL_POS",
        81 => "ADD_REL_POS",
        82 => "RWKV_WKV6",
        83 => "GATED_LINEAR_ATTN",
        84 => "RWKV_WKV7",
        85 => "SOLVE_TRI",
        86 => "GATED_DELTA_NET",
        87 => "LIGHTNING_INDEXER",
        88 => "DSV4_HC_COMB",
        89 => "DSV4_HC_PRE",
        90 => "DSV4_HC_POST",
        91 => "UNARY",
        96 => "CROSS_ENTROPY_LOSS",
        97 => "CROSS_ENTROPY_LOSS_BACK",
        98 => "OPT_STEP_ADAMW",
        99 => "OPT_STEP_SGD",
        100 => "GLU",
        101 => "GATED_DELTA_NET_REPLAY",
        _ => "UNKNOWN",
    }
}

pub(super) fn unary_op_name(uop: i32) -> &'static str {
    match uop {
        0 => "ABS",
        1 => "SGN",
        2 => "NEG",
        3 => "STEP",
        4 => "TANH",
        5 => "ELU",
        6 => "RELU",
        7 => "SIGMOID",
        8 => "GELU",
        9 => "GELU_QUICK",
        10 => "SILU",
        11 => "HARDSWISH",
        12 => "HARDSIGMOID",
        13 => "EXP",
        14 => "EXPM1",
        15 => "SOFTPLUS",
        16 => "GELU_ERF",
        17 => "XIELU",
        18 => "FLOOR",
        19 => "CEIL",
        20 => "ROUND",
        21 => "TRUNC",
        _ => "UNKNOWN",
    }
}

pub(super) fn glu_op_name(gop: i32) -> &'static str {
    match gop {
        0 => "REGLU",
        1 => "GEGLU",
        2 => "SWIGLU",
        3 => "SWIGLU_OAI",
        4 => "GEGLU_ERF",
        5 => "GEGLU_QUICK",
        _ => "UNKNOWN",
    }
}

#[derive(Clone)]
pub(super) struct TensorMeta {
    pub(super) id: u32,
    pub(super) aliases: Vec<u32>,
    pub(super) kind: i32,
    pub(super) ne: [i64; GGML_MAX_DIMS],
    pub(super) nb: [u64; GGML_MAX_DIMS],
    pub(super) op: i32,
    pub(super) flags: i32,
    pub(super) name: String,
    pub(super) view_source: Option<u32>,
    pub(super) view_offset: u64,
    pub(super) sources: [Option<u32>; GGML_MAX_SRC],
    pub(super) op_params: [u8; GGML_MAX_OP_PARAMS],
    pub(super) bytes: u64,
    pub(super) buffer_id: Option<u32>,
    pub(super) buffer_offset: u64,
}

impl TensorMeta {
    pub(super) fn new_base(id: u32, kind: i32, ne: [i64; GGML_MAX_DIMS]) -> Result<Self> {
        let nb = contiguous_strides(kind, ne)?;
        let bytes = compute_nbytes(kind, ne, nb)?;
        Ok(Self {
            id,
            aliases: vec![id],
            kind,
            ne,
            nb,
            op: OP_NONE,
            flags: 0,
            name: String::new(),
            view_source: None,
            view_offset: 0,
            sources: [None; GGML_MAX_SRC],
            op_params: [0_u8; GGML_MAX_OP_PARAMS],
            bytes,
            buffer_id: None,
            buffer_offset: 0,
        })
    }

    pub(super) fn nelements(&self) -> i64 {
        self.ne.iter().product()
    }

    pub(super) fn nrows(&self) -> i64 {
        self.ne[1] * self.ne[2] * self.ne[3]
    }

    pub(super) fn op_param_i32(&self, idx: usize) -> i32 {
        let off = idx * 4;
        if let Some(slice) = self.op_params.get(off..off + 4)
            && let Ok(arr) = slice.try_into()
        {
            return i32::from_le_bytes(arr);
        }
        0
    }

    pub(super) fn op_param_f32(&self, idx: usize) -> f32 {
        let off = idx * 4;
        if let Some(slice) = self.op_params.get(off..off + 4)
            && let Ok(arr) = slice.try_into()
        {
            return f32::from_le_bytes(arr);
        }
        0.0
    }

    pub(super) fn set_op_param_i32(&mut self, idx: usize, val: i32) {
        let off = idx * 4;
        if let Some(slice) = self.op_params.get_mut(off..off + 4) {
            slice.copy_from_slice(&val.to_le_bytes());
        }
    }

    pub(super) fn set_op_param_f32(&mut self, idx: usize, val: f32) {
        let off = idx * 4;
        if let Some(slice) = self.op_params.get_mut(off..off + 4) {
            slice.copy_from_slice(&val.to_le_bytes());
        }
    }

    pub(super) fn op_description(&self) -> String {
        if self.op == OP_UNARY {
            unary_op_name(self.op_param_i32(0)).to_owned()
        } else if self.op == OP_GLU {
            glu_op_name(self.op_param_i32(0)).to_owned()
        } else {
            op_name(self.op).to_owned()
        }
    }

    pub(super) fn refresh_bytes(&mut self) -> Result<()> {
        self.bytes = compute_nbytes(self.kind, self.ne, self.nb)?;
        Ok(())
    }

    pub(super) fn to_snapshot_json(&self) -> Value {
        json!({
            "Id": self.id,
            "Aliases": self.aliases,
            "Type": self.kind,
            "NE": self.ne,
            "NB": self.nb,
            "Op": self.op,
            "Flags": self.flags,
            "Name": self.name,
            "ViewSource": self.view_source,
            "ViewOffset": self.view_offset,
            "Sources": self.sources,
            "OpParamsHex": encode_hex(&self.op_params),
            "NBytes": self.bytes,
            "NElements": self.nelements(),
            "NRows": self.nrows(),
            "BlockSize": blck_size(self.kind),
            "TypeSize": type_size(self.kind),
            "OpDescription": self.op_description(),
        })
    }
}

pub(super) fn encode_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        out.push(char::from(HEX[usize::from(byte >> 4)]));
        out.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    out
}

pub(super) fn decode_hex(text: &str) -> Result<Vec<u8>> {
    let raw = text.as_bytes();
    if !raw.len().is_multiple_of(2) {
        return Err(invalid("hex string must have even length"));
    }
    let hex_nibble = |ch: u8| -> Result<u8> {
        match ch {
            b'0'..=b'9' => Ok(ch - b'0'),
            b'a'..=b'f' => Ok(ch - b'a' + 10),
            b'A'..=b'F' => Ok(ch - b'A' + 10),
            _ => Err(invalid("invalid hex digit")),
        }
    };
    let mut out = Vec::with_capacity(raw.len() / 2);
    for pair in raw.as_chunks::<2>().0 {
        out.push((hex_nibble(pair[0])? << 4) | hex_nibble(pair[1])?);
    }
    Ok(out)
}

pub(super) fn contiguous_strides(kind: i32, ne: [i64; 4]) -> Result<[u64; 4]> {
    let ts = type_size(kind);
    let bs = u64::try_from(blck_size(kind))?;
    if !u64::try_from(ne[0])?.is_multiple_of(bs) {
        return Err(invalid("tensor ne[0] not divisible by block size"));
    }
    let nb0 = ts;
    let nb1 = ts
        .checked_mul(u64::try_from(ne[0])? / bs)
        .ok_or_else(|| invalid("nb1 stride overflow"))?;
    let nb2 = nb1
        .checked_mul(u64::try_from(ne[1])?)
        .ok_or_else(|| invalid("nb2 stride overflow"))?;
    let nb3 = nb2
        .checked_mul(u64::try_from(ne[2])?)
        .ok_or_else(|| invalid("nb3 stride overflow"))?;
    Ok([nb0, nb1, nb2, nb3])
}

pub(super) fn compute_nbytes(kind: i32, ne: [i64; 4], nb: [u64; 4]) -> Result<u64> {
    let bs = blck_size(kind);
    let ts = type_size(kind);
    let n0 = u64::try_from(ne[0])?;
    let n1 = u64::try_from(ne[1])?;
    let n2 = u64::try_from(ne[2])?;
    let n3 = u64::try_from(ne[3])?;
    if n0 == 0 || n1 == 0 || n2 == 0 || n3 == 0 {
        return Ok(0);
    }
    let mut b = ((n0 / u64::try_from(bs)?) - 1) * nb[0] + ts;
    b += (n1 - 1) * nb[1];
    b += (n2 - 1) * nb[2];
    b += (n3 - 1) * nb[3];
    Ok(b)
}

pub(super) fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exp = u32::from((bits >> 10) & 31);
    let mut mant = u32::from(bits & 1023);
    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign);
        }
        let mut e = 113_u32;
        while mant & 1024 == 0 {
            mant <<= 1;
            e -= 1;
        }
        return f32::from_bits(sign | (e << 23) | ((mant & 1023) << 13));
    }
    if exp == 31 {
        return f32::from_bits(sign | 0x7f80_0000 | (mant << 13));
    }
    f32::from_bits(sign | ((exp + 112) << 23) | (mant << 13))
}

pub(super) fn f32_to_f16(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = u16::try_from((bits >> 16) & 0x8000).unwrap_or(0);
    let exp = i32::try_from((bits >> 23) & 0xff).unwrap_or(0) - 127;
    let mant = bits & 0x007f_ffff;
    if exp == 128 {
        let frac = if mant == 0 {
            0
        } else {
            u16::try_from((mant >> 13) | 1).unwrap_or(1)
        };
        return sign | 0x7c00 | frac;
    }
    if exp > 15 {
        return sign | 0x7c00;
    }
    if exp >= -14 {
        let round = (mant + 0x1000) >> 13;
        let he = u16::try_from(exp + 15).unwrap_or(0);
        let hr = u16::try_from(round).unwrap_or(0);
        return sign + (he << 10) + hr;
    }
    if exp < -24 {
        return sign;
    }
    let full = mant | 0x0080_0000;
    let shift = u32::try_from(-14 - exp + 13).unwrap_or(24);
    let rounded = full.checked_shr(shift).unwrap_or(0);
    sign | u16::try_from(rounded).unwrap_or(0)
}

pub(super) fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

pub(super) fn f32_to_bf16(val: f32) -> u16 {
    let bits = val.to_bits();
    if val.is_nan() {
        return u16::try_from(bits >> 16).unwrap_or(0x7fc0) | 0x0040;
    }
    let round = bits.wrapping_add(0x7fff + ((bits >> 16) & 1));
    u16::try_from(round >> 16).unwrap_or(0)
}

pub(super) fn f64_to_f32(val: f64) -> f32 {
    let bits = val.to_bits();
    let sign = u32::from(bits >> 63 != 0) << 31;
    if val.is_nan() {
        let payload = u32::try_from((bits >> 29) & 0x003f_ffff).unwrap_or(0);
        return f32::from_bits(sign | 0x7fc0_0000 | payload);
    }
    let raw_exp = (bits >> 52) & 0x7ff;
    if raw_exp == 0x7ff {
        return f32::from_bits(sign | 0x7f80_0000);
    }
    let abs_bits = bits & !(1_u64 << 63);
    if abs_bits == 0 {
        return f32::from_bits(sign);
    }
    let exp = i32::try_from(raw_exp).unwrap_or(0) - 1023;
    let frac = bits & 0x000f_ffff_ffff_ffff;
    if exp >= -126 {
        let mant = frac | 0x0010_0000_0000_0000;
        let keep = u32::try_from(mant >> 29).unwrap_or(0);
        let rem = mant & ((1_u64 << 29) - 1);
        let half = 1_u64 << 28;
        let round_up = rem > half || (rem == half && (keep & 1) == 1);
        let mut rounded = keep + u32::from(round_up);
        let mut f32_exp = exp + 127;
        if rounded >= 0x0100_0000 {
            rounded >>= 1;
            f32_exp += 1;
        }
        if f32_exp >= 255 {
            return f32::from_bits(sign | 0x7f80_0000);
        }
        let exp_bits = u32::try_from(f32_exp).unwrap_or(0) << 23;
        return f32::from_bits(sign | exp_bits | (rounded & 0x007f_ffff));
    }
    if exp < -150 {
        return f32::from_bits(sign);
    }
    let mant = frac | 0x0010_0000_0000_0000;
    let drop_bits = u32::try_from(-126 - exp + 29).unwrap_or(53);
    let keep = u32::try_from(mant.checked_shr(drop_bits).unwrap_or(0)).unwrap_or(0);
    let mask = 1_u64.checked_shl(drop_bits).map_or(u64::MAX, |v| v - 1);
    let rem = mant & mask;
    let half = drop_bits
        .checked_sub(1)
        .and_then(|s| 1_u64.checked_shl(s))
        .unwrap_or(0);
    let round_up = rem > half || (rem == half && (keep & 1) == 1);
    let rounded = keep + u32::from(round_up);
    f32::from_bits(sign | rounded)
}

pub(super) fn i64_to_f32(val: i64) -> f32 {
    if val == 0 {
        return 0.0;
    }
    let sign = u32::from(val < 0) << 31;
    let mag = val.unsigned_abs();
    let msb = mag.ilog2();
    if msb <= 23 {
        let mant = u32::try_from(mag << (23 - msb)).unwrap_or(0) & 0x007f_ffff;
        let exp = (msb + 127) << 23;
        return f32::from_bits(sign | exp | mant);
    }
    let drop = msb - 23;
    let keep = u32::try_from(mag >> drop).unwrap_or(0);
    let mask = (1_u64 << drop) - 1;
    let rem = mag & mask;
    let half = 1_u64 << (drop - 1);
    let round_up = rem > half || (rem == half && (keep & 1) == 1);
    let mut rounded = keep + u32::from(round_up);
    let mut exp = msb + 127;
    if rounded >= 0x0100_0000 {
        rounded >>= 1;
        exp += 1;
    }
    f32::from_bits(sign | (exp << 23) | (rounded & 0x007f_ffff))
}

pub(super) fn f32_to_i64_trunc(val: f32) -> i64 {
    if !val.is_finite() {
        return 0;
    }
    let t = val.trunc();
    let bits = t.to_bits();
    let neg = (bits >> 31) != 0;
    let exp = i32::try_from((bits >> 23) & 0xff).unwrap_or(0) - 127;
    if exp < 0 {
        return 0;
    }
    if exp >= 63 {
        return if neg { i64::MIN } else { i64::MAX };
    }
    let mant = u64::from((bits & 0x007f_ffff) | 0x0080_0000);
    let mag = if exp >= 23 {
        mant << u32::try_from(exp - 23).unwrap_or(0)
    } else {
        mant >> u32::try_from(23 - exp).unwrap_or(0)
    };
    let signed = i64::try_from(mag).unwrap_or(i64::MAX);
    if neg { -signed } else { signed }
}

pub(super) fn read_unquant_elem_f32(kind: i32, raw: &[u8]) -> Result<f32> {
    match kind {
        TYPE_F32 => Ok(f32::from_le_bytes(raw[0..4].try_into()?)),
        TYPE_F16 => Ok(f16_to_f32(u16::from_le_bytes(raw[0..2].try_into()?))),
        TYPE_BF16 => Ok(bf16_to_f32(u16::from_le_bytes(raw[0..2].try_into()?))),
        TYPE_F64 => Ok(f64_to_f32(f64::from_le_bytes(raw[0..8].try_into()?))),
        TYPE_I8 => Ok(f32::from(i8::from_le_bytes([raw[0]]))),
        TYPE_I16 => Ok(f32::from(i16::from_le_bytes(raw[0..2].try_into()?))),
        TYPE_I32 => Ok(f64_to_f32(f64::from(i32::from_le_bytes(
            raw[0..4].try_into()?,
        )))),
        TYPE_I64 => Ok(i64_to_f32(i64::from_le_bytes(raw[0..8].try_into()?))),
        _ => Err(invalid(format!(
            "cannot read element of quantized/unknown type {kind}"
        ))),
    }
}

pub(super) fn write_unquant_elem_f32(kind: i32, dst: &mut [u8], val: f32) -> Result<()> {
    match kind {
        TYPE_F32 => dst[0..4].copy_from_slice(&val.to_le_bytes()),
        TYPE_F16 => dst[0..2].copy_from_slice(&f32_to_f16(val).to_le_bytes()),
        TYPE_BF16 => dst[0..2].copy_from_slice(&f32_to_bf16(val).to_le_bytes()),
        TYPE_F64 => dst[0..8].copy_from_slice(&f64::from(val).to_le_bytes()),
        TYPE_I8 => {
            let q =
                i8::try_from(f32_to_i64_trunc(val).clamp(i64::from(i8::MIN), i64::from(i8::MAX)))
                    .unwrap_or(0);
            dst[0] = q.to_le_bytes()[0];
        }
        TYPE_I16 => {
            let q = i16::try_from(
                f32_to_i64_trunc(val).clamp(i64::from(i16::MIN), i64::from(i16::MAX)),
            )
            .unwrap_or(0);
            dst[0..2].copy_from_slice(&q.to_le_bytes());
        }
        TYPE_I32 => {
            let q = i32::try_from(
                f32_to_i64_trunc(val).clamp(i64::from(i32::MIN), i64::from(i32::MAX)),
            )
            .unwrap_or(0);
            dst[0..4].copy_from_slice(&q.to_le_bytes());
        }
        TYPE_I64 => {
            let q = f32_to_i64_trunc(val);
            dst[0..8].copy_from_slice(&q.to_le_bytes());
        }
        _ => {
            return Err(invalid(format!(
                "cannot write element of quantized/unknown type {kind}"
            )));
        }
    }
    Ok(())
}

pub(super) fn i32_to_f32(val: i32) -> f32 {
    f64_to_f32(f64::from(val))
}

pub(super) fn usize_to_f32(val: usize) -> f32 {
    i64::try_from(val).map_or(f32::MAX, i64_to_f32)
}

pub(super) fn read_i32_elem(kind: i32, raw: &[u8]) -> Result<i32> {
    match kind {
        TYPE_I32 => Ok(i32::from_le_bytes(raw[0..4].try_into()?)),
        TYPE_I64 => Ok(i32::try_from(i64::from_le_bytes(raw[0..8].try_into()?))?),
        TYPE_I16 => Ok(i32::from(i16::from_le_bytes(raw[0..2].try_into()?))),
        TYPE_I8 => Ok(i32::from(i8::from_le_bytes([raw[0]]))),
        _ => Err(invalid(format!("expected integer tensor type, got {kind}"))),
    }
}

pub(super) fn write_i32_elem(kind: i32, dst: &mut [u8], val: i32) -> Result<()> {
    match kind {
        TYPE_I32 => dst[0..4].copy_from_slice(&val.to_le_bytes()),
        TYPE_I64 => dst[0..8].copy_from_slice(&i64::from(val).to_le_bytes()),
        _ => return Err(invalid(format!("cannot write i32 to tensor type {kind}"))),
    }
    Ok(())
}
