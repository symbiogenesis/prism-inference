//! Clean-room tokenization and detokenization algorithms (`SPM`, `BPE`, `WPM`, `UGM`, `RWKV`, `PLaMo2`).
//!
//! No `bonsai_prism_runtime` or external tokenizer reference crates are imported or called.

use std::cmp::Reverse;
use std::collections::{BTreeMap, BinaryHeap, HashMap};

use super::super::{Result, invalid};
use super::gguf::MetaValue;
use super::pretokenizer::{
    build_gpt_byte_alphabet, build_gpt_char_to_byte, compile_pretokenizer, lookup_gpt_byte,
    split_text,
};
use fancy_regex::Regex;

const MAX_TOKENS_OUTPUT: usize = 262_144;
const MAX_VOCAB_SIZE: usize = 1_000_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum VocabType {
    None = 0,
    Spm = 1,
    Bpe = 2,
    Wpm = 3,
    Ugm = 4,
    Rwkv = 5,
    Plamo2 = 6,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum TokenType {
    Undefined = 0,
    Normal = 1,
    Unknown = 2,
    Control = 3,
    UserDefined = 4,
    Unused = 5,
    Byte = 6,
}

impl TokenType {
    const fn from_raw(value: i64) -> Self {
        match value {
            1 => Self::Normal,
            2 => Self::Unknown,
            3 => Self::Control,
            4 => Self::UserDefined,
            5 => Self::Unused,
            6 => Self::Byte,
            _ => Self::Undefined,
        }
    }

    const fn as_i32(self) -> i32 {
        match self {
            Self::Undefined => 0,
            Self::Normal => 1,
            Self::Unknown => 2,
            Self::Control => 3,
            Self::UserDefined => 4,
            Self::Unused => 5,
            Self::Byte => 6,
        }
    }
}

#[derive(Clone)]
struct TokenEntry {
    raw: Vec<u8>,
    text: String,
    score: f32,
    kind: TokenType,
}

type SpmHeap = BinaryHeap<(ScoreKey, Reverse<usize>, usize, usize, u64, u64)>;
type BpeHeap = BinaryHeap<Reverse<(usize, usize, usize, usize, u64, u64)>>;

/// Clean-room independent GGUF vocabulary model and tokenizer.
#[expect(
    clippy::struct_excessive_bools,
    reason = "Independent GGUF tokenizer configuration booleans map 1:1 to GGUF metadata keys."
)]
#[expect(
    dead_code,
    reason = "Session-level special token IDs, metadata fields, and token helpers are also consumed by subsequent model-session retirement steps."
)]
pub(in super::super) struct ReferenceVocab {
    vocab_type: VocabType,
    pre_type: String,
    entries: Vec<TokenEntry>,
    token_to_id: HashMap<Vec<u8>, i32>,
    merges: HashMap<(Vec<u8>, Vec<u8>), usize>,
    pre_patterns: Vec<Regex>,
    gpt_alphabet: [String; 256],
    gpt_char_to_byte: Vec<(char, u8)>,
    byte_tokens: [Option<i32>; 256],
    specials: Vec<(Vec<u8>, i32, bool)>,
    pub(super) bos: Option<i32>,
    pub(super) eos: Option<i32>,
    pub(super) eot: Option<i32>,
    pub(super) eom: Option<i32>,
    pub(super) unk: Option<i32>,
    pub(super) sep: Option<i32>,
    pub(super) nl: Option<i32>,
    pub(super) pad: Option<i32>,
    pub(super) cls: Option<i32>,
    pub(super) mask: Option<i32>,
    pub(super) add_bos: bool,
    pub(super) add_eos: bool,
    pub(super) add_sep: bool,
    add_space_prefix: bool,
    escape_whitespaces: bool,
    treat_whitespace_as_suffix: bool,
    ignore_merges: bool,
    lowercase_normalizer: bool,
    strip_accents_normalizer: bool,
    pub(super) chat_template: Option<String>,
}

fn opt_token_id(meta: &BTreeMap<String, MetaValue>, key: &str, count: usize) -> Option<i32> {
    meta.get(key)
        .and_then(MetaValue::as_token_id)
        .filter(|id| usize::try_from(*id).is_ok_and(|idx| idx < count))
        .and_then(|id| i32::try_from(id).ok())
}

fn opt_bool(meta: &BTreeMap<String, MetaValue>, key: &str) -> Option<bool> {
    meta.get(key).and_then(MetaValue::as_bool)
}

fn parse_vocab_type(meta: &BTreeMap<String, MetaValue>) -> Result<VocabType> {
    match meta
        .get("tokenizer.ggml.model")
        .and_then(MetaValue::as_str_lossy)
        .as_deref()
    {
        Some("no_vocab" | "none") => Ok(VocabType::None),
        Some("llama" | "replit") | None => Ok(VocabType::Spm),
        Some("gpt2" | "bpe") => Ok(VocabType::Bpe),
        Some("bert") => Ok(VocabType::Wpm),
        Some("t5" | "ugm") => Ok(VocabType::Ugm),
        Some("rwkv") => Ok(VocabType::Rwkv),
        Some("plamo2") => Ok(VocabType::Plamo2),
        Some(other) => Err(invalid(format!("unsupported GGUF tokenizer model {other}"))),
    }
}

fn parse_byte_token_hex(raw: &[u8]) -> Option<u8> {
    if raw.len() == 6 && raw.starts_with(b"<0x") && raw[5] == b'>' {
        let s = std::str::from_utf8(&raw[3..5]).ok()?;
        u8::from_str_radix(s, 16).ok()
    } else {
        None
    }
}

impl ReferenceVocab {
    pub(super) fn from_metadata(meta: &BTreeMap<String, MetaValue>) -> Result<Self> {
        let vocab_type = parse_vocab_type(meta)?;
        let Some(MetaValue::BytesArray(raw_tokens)) = meta.get("tokenizer.ggml.tokens") else {
            return Err(invalid("vocabulary requires a STRING token array"));
        };
        if raw_tokens.is_empty() || raw_tokens.len() > MAX_VOCAB_SIZE {
            return Err(invalid("vocabulary token count outside bounded range"));
        }
        let count = raw_tokens.len();
        let scores = match meta.get("tokenizer.ggml.scores") {
            Some(MetaValue::FloatArray(values)) if values.len() == count => Some(values),
            Some(MetaValue::FloatArray(_)) => {
                return Err(invalid("vocabulary scores length differs from tokens"));
            }
            _ => None,
        };
        let types = match meta.get("tokenizer.ggml.token_type") {
            Some(MetaValue::IntArray(values)) if values.len() == count => Some(values),
            Some(MetaValue::UIntArray(values)) if values.len() == count => {
                let converted = values
                    .iter()
                    .map(|&v| i64::try_from(v))
                    .collect::<std::result::Result<Vec<_>, _>>()?;
                return Self::from_metadata_with_types(
                    meta,
                    vocab_type,
                    raw_tokens,
                    scores,
                    Some(&converted),
                );
            }
            Some(MetaValue::IntArray(_) | MetaValue::UIntArray(_)) => {
                return Err(invalid("vocabulary token_type length differs from tokens"));
            }
            _ => None,
        };
        Self::from_metadata_with_types(
            meta,
            vocab_type,
            raw_tokens,
            scores,
            types.map(Vec::as_slice),
        )
    }

    #[expect(
        clippy::too_many_lines,
        reason = "Vocab table construction and special/byte fallback indexing are linear and self-contained."
    )]
    fn from_metadata_with_types(
        meta: &BTreeMap<String, MetaValue>,
        vocab_type: VocabType,
        raw_tokens: &[Vec<u8>],
        scores: Option<&Vec<f32>>,
        types: Option<&[i64]>,
    ) -> Result<Self> {
        let count = raw_tokens.len();
        let mut entries = Vec::with_capacity(count);
        let mut token_to_id = HashMap::with_capacity(count);
        let mut byte_tokens = [None; 256];
        let mut specials = Vec::new();
        for (idx, raw) in raw_tokens.iter().enumerate() {
            let id = i32::try_from(idx)?;
            let score = scores.map_or(0.0, |v| v[idx]);
            let kind = types.map_or(TokenType::Normal, |v| TokenType::from_raw(v[idx]));
            let text = String::from_utf8_lossy(raw).into_owned();
            if kind == TokenType::Byte
                && let Some(byte) = parse_byte_token_hex(raw)
            {
                byte_tokens[usize::from(byte)] = Some(id);
            }
            if matches!(kind, TokenType::Control | TokenType::UserDefined) && !raw.is_empty() {
                specials.push((raw.clone(), id, kind == TokenType::Control));
            }
            token_to_id.entry(raw.clone()).or_insert(id);
            entries.push(TokenEntry {
                raw: raw.clone(),
                text,
                score,
                kind,
            });
        }
        // Sort longest special piece first for deterministic greedy matching at each offset.
        specials.sort_by(|a, b| b.0.len().cmp(&a.0.len()).then_with(|| a.1.cmp(&b.1)));

        let mut merges = HashMap::new();
        if let Some(MetaValue::BytesArray(raw_merges)) = meta.get("tokenizer.ggml.merges") {
            merges.reserve(raw_merges.len());
            for (rank, raw) in raw_merges.iter().enumerate() {
                let s = std::str::from_utf8(raw)
                    .map_err(|_| invalid("vocabulary merge rule is not valid UTF-8"))?;
                if let Some((left, right)) = s.split_once(' ') {
                    merges
                        .entry((left.as_bytes().to_vec(), right.as_bytes().to_vec()))
                        .or_insert(rank);
                }
            }
        }

        let pre_type = meta
            .get("tokenizer.ggml.pre")
            .and_then(MetaValue::as_str_lossy)
            .unwrap_or_else(|| "default".to_owned());
        let pre_patterns = if vocab_type == VocabType::Bpe {
            compile_pretokenizer(&pre_type)?
        } else {
            Vec::new()
        };
        let gpt_alphabet = build_gpt_byte_alphabet()?;
        let gpt_char_to_byte = build_gpt_char_to_byte(&gpt_alphabet);
        if vocab_type == VocabType::Bpe {
            for byte in 0..=u8::MAX {
                let key = gpt_alphabet[usize::from(byte)].as_bytes();
                if let Some(&id) = token_to_id.get(key) {
                    byte_tokens[usize::from(byte)].get_or_insert(id);
                }
            }
        }

        let bos = opt_token_id(meta, "tokenizer.ggml.bos_token_id", count)
            .or_else(|| opt_token_id(meta, "tokenizer.ggml.cls_token_id", count));
        let eos = opt_token_id(meta, "tokenizer.ggml.eos_token_id", count);
        let eot = opt_token_id(meta, "tokenizer.ggml.eot_token_id", count);
        let eom = opt_token_id(meta, "tokenizer.ggml.eom_token_id", count);
        let unk = opt_token_id(meta, "tokenizer.ggml.unknown_token_id", count).or_else(|| {
            entries
                .iter()
                .position(|e| e.kind == TokenType::Unknown)
                .and_then(|i| i32::try_from(i).ok())
        });
        let sep = opt_token_id(meta, "tokenizer.ggml.seperator_token_id", count)
            .or_else(|| opt_token_id(meta, "tokenizer.ggml.separator_token_id", count));
        let nl = opt_token_id(meta, "tokenizer.ggml.nl_token_id", count);
        let pad = opt_token_id(meta, "tokenizer.ggml.padding_token_id", count);
        let cls = opt_token_id(meta, "tokenizer.ggml.cls_token_id", count);
        let mask = opt_token_id(meta, "tokenizer.ggml.mask_token_id", count);

        let is_llama3_pre = matches!(pre_type.as_str(), "llama3" | "llama-v3" | "llama-bpe");
        let prepend_bos = opt_bool(meta, "tokenizer.ggml.add_bos_token").unwrap_or(
            matches!(vocab_type, VocabType::Spm | VocabType::Ugm | VocabType::Wpm) || is_llama3_pre,
        );
        let append_eos = opt_bool(meta, "tokenizer.ggml.add_eos_token").unwrap_or(false);
        let add_sep = opt_bool(meta, "tokenizer.ggml.add_sep_token")
            .unwrap_or_else(|| vocab_type == VocabType::Wpm && sep.is_some());
        let add_space_prefix = opt_bool(meta, "tokenizer.ggml.add_space_prefix")
            .unwrap_or(matches!(vocab_type, VocabType::Spm | VocabType::Ugm));
        let escape_whitespaces = opt_bool(meta, "tokenizer.ggml.escape_whitespaces")
            .unwrap_or(matches!(vocab_type, VocabType::Spm | VocabType::Ugm));
        let treat_whitespace_as_suffix =
            opt_bool(meta, "tokenizer.ggml.treat_whitespace_as_suffix").unwrap_or(false);
        // llama3-cleanup.patch disables clean_spaces for llama3/llama-v3/llama-bpe while keeping ignore_merges.
        let ignore_merges = opt_bool(meta, "tokenizer.ggml.ignore_merges").unwrap_or(is_llama3_pre);
        let lowercase_normalizer = opt_bool(meta, "tokenizer.ggml.normalizer.lowercase")
            .unwrap_or(vocab_type == VocabType::Wpm);
        let strip_accents_normalizer =
            opt_bool(meta, "tokenizer.ggml.normalizer.strip_accents").unwrap_or(false);
        let chat_template = meta
            .get("tokenizer.chat_template")
            .and_then(MetaValue::as_str_lossy);

        Ok(Self {
            vocab_type,
            pre_type,
            entries,
            token_to_id,
            merges,
            pre_patterns,
            gpt_alphabet,
            gpt_char_to_byte,
            byte_tokens,
            specials,
            bos,
            eos,
            eot,
            eom,
            unk,
            sep,
            nl,
            pad,
            cls,
            mask,
            add_bos: prepend_bos,
            add_eos: append_eos,
            add_sep,
            add_space_prefix,
            escape_whitespaces,
            treat_whitespace_as_suffix,
            ignore_merges,
            lowercase_normalizer,
            strip_accents_normalizer,
            chat_template,
        })
    }

    #[expect(
        dead_code,
        reason = "Session metadata helpers are used across subsequent tokenizer/llguidance/model session adapters."
    )]
    pub(super) const fn vocab_type(&self) -> VocabType {
        self.vocab_type
    }

    #[expect(
        dead_code,
        reason = "Session metadata helpers are used across subsequent tokenizer/llguidance/model session adapters."
    )]
    pub(super) const fn vocab_size(&self) -> usize {
        self.entries.len()
    }

    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "Session metadata helpers are used across subsequent tokenizer/llguidance/model session adapters."
        )
    )]
    pub(super) fn is_control(&self, token: i32) -> bool {
        usize::try_from(token)
            .ok()
            .and_then(|idx| self.entries.get(idx))
            .is_some_and(|entry| entry.kind == TokenType::Control)
    }

    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "Session metadata helpers are used across subsequent tokenizer/llguidance/model session adapters."
        )
    )]
    pub(super) fn is_eog(&self, token: i32) -> bool {
        Some(token) == self.eos
            || Some(token) == self.eot
            || Some(token) == self.eom
            || self.is_control_eog(token)
    }

    fn is_control_eog(&self, token: i32) -> bool {
        usize::try_from(token)
            .ok()
            .and_then(|idx| self.entries.get(idx))
            .is_some_and(|entry| {
                entry.kind == TokenType::Control
                    && matches!(
                        entry.text.as_str(),
                        "</s>"
                            | "<|endoftext|>"
                            | "<|eot_id|>"
                            | "<|im_end|>"
                            | "<end_of_turn>"
                            | "<|end|>"
                    )
            })
    }

    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "Session metadata helpers are used across subsequent tokenizer/llguidance/model session adapters."
        )
    )]
    pub(super) fn token_type_raw(&self, token: i32) -> Option<i32> {
        usize::try_from(token)
            .ok()
            .and_then(|idx| self.entries.get(idx))
            .map(|entry| entry.kind.as_i32())
    }

    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "Session metadata helpers are used across subsequent tokenizer/llguidance/model session adapters."
        )
    )]
    pub(super) fn token_text_raw(&self, token: i32) -> Option<&[u8]> {
        usize::try_from(token)
            .ok()
            .and_then(|idx| self.entries.get(idx))
            .map(|entry| entry.raw.as_slice())
    }

    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "Session metadata helpers are used across subsequent tokenizer/llguidance/model session adapters."
        )
    )]
    pub(super) fn token_attr_flags(&self, token: i32) -> Option<i32> {
        let entry = usize::try_from(token)
            .ok()
            .and_then(|idx| self.entries.get(idx))?;
        let mut mask = 0_i32;
        match entry.kind {
            TokenType::Undefined => mask |= 1,
            TokenType::Unknown => mask |= 2,
            TokenType::Control => mask |= 4,
            TokenType::Normal => mask |= 8,
            TokenType::UserDefined => mask |= 16,
            TokenType::Byte => mask |= 32,
            TokenType::Unused => mask |= 64,
        }
        Some(mask)
    }

    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "Session piece helpers are used across subsequent tokenizer/llguidance/model session adapters."
        )
    )]
    pub(super) fn token_to_piece(&self, token: i32, lstrip: i32, special: bool) -> Result<Vec<u8>> {
        let idx = usize::try_from(token).map_err(|_| invalid("negative token ID"))?;
        let entry = self
            .entries
            .get(idx)
            .ok_or_else(|| invalid("token ID outside vocabulary"))?;
        let bytes = self.raw_piece_bytes(entry, special);
        if lstrip > 0 {
            let skip = usize::try_from(lstrip).unwrap_or(0);
            let mut trimmed = bytes.as_slice();
            for _ in 0..skip {
                if let Some(rest) = trimmed.strip_prefix(b" ") {
                    trimmed = rest;
                } else {
                    break;
                }
            }
            return Ok(trimmed.to_vec());
        }
        Ok(bytes)
    }

    fn raw_piece_bytes(&self, entry: &TokenEntry, special: bool) -> Vec<u8> {
        match entry.kind {
            TokenType::Control | TokenType::Unused | TokenType::Unknown => {
                if special {
                    entry.raw.clone()
                } else {
                    Vec::new()
                }
            }
            TokenType::Byte => parse_byte_token_hex(&entry.raw)
                .map_or_else(|| entry.raw.clone(), |byte| vec![byte]),
            TokenType::Normal | TokenType::UserDefined | TokenType::Undefined => {
                match self.vocab_type {
                    VocabType::Spm | VocabType::Ugm | VocabType::Wpm => {
                        if self.escape_whitespaces || self.vocab_type == VocabType::Wpm {
                            replace_bytes(&entry.raw, "▁".as_bytes(), b" ")
                        } else {
                            entry.raw.clone()
                        }
                    }
                    VocabType::Bpe => {
                        if entry.kind == TokenType::UserDefined {
                            return entry.raw.clone();
                        }
                        if let Ok(s) = std::str::from_utf8(&entry.raw) {
                            let mut out = Vec::with_capacity(s.len());
                            let mut all_gpt = true;
                            for ch in s.chars() {
                                if let Some(byte) = lookup_gpt_byte(&self.gpt_char_to_byte, ch) {
                                    out.push(byte);
                                } else {
                                    all_gpt = false;
                                    break;
                                }
                            }
                            if all_gpt {
                                return out;
                            }
                        }
                        entry.raw.clone()
                    }
                    VocabType::None | VocabType::Rwkv | VocabType::Plamo2 => entry.raw.clone(),
                }
            }
        }
    }

    pub(super) fn detokenize(
        &self,
        tokens: &[i32],
        remove_special: bool,
        unparse_special: bool,
    ) -> Result<Vec<u8>> {
        let mut out = Vec::new();
        let mut leading_controls_done = false;
        let mut first_non_control_byte_idx = None;
        for &token in tokens {
            let idx = usize::try_from(token).map_err(|_| invalid("negative token ID"))?;
            let entry = self
                .entries
                .get(idx)
                .ok_or_else(|| invalid("detokenizer token outside vocabulary"))?;
            if remove_special && entry.kind == TokenType::Control {
                continue;
            }
            let piece = self.raw_piece_bytes(entry, unparse_special);
            if !leading_controls_done && entry.kind != TokenType::Control {
                leading_controls_done = true;
                if first_non_control_byte_idx.is_none() {
                    first_non_control_byte_idx = Some(out.len());
                }
            }
            out.extend_from_slice(&piece);
            if out.len() > 4 * 1024 * 1024 {
                return Err(invalid("detokenized output exceeds bound"));
            }
        }
        if matches!(self.vocab_type, VocabType::Spm | VocabType::Ugm)
            && self.add_space_prefix
            && !self.treat_whitespace_as_suffix
            && let Some(idx) = first_non_control_byte_idx
            && out.get(idx) == Some(&b' ')
        {
            out.remove(idx);
        }
        if matches!(self.vocab_type, VocabType::Spm | VocabType::Ugm)
            && self.add_space_prefix
            && self.treat_whitespace_as_suffix
            && out.last() == Some(&b' ')
        {
            out.pop();
        }
        Ok(out)
    }

    pub(super) fn tokenize(
        &self,
        input: &[u8],
        add_special: bool,
        parse_special: bool,
    ) -> Result<Vec<i32>> {
        if input.len() > 1024 * 1024 {
            return Err(invalid("tokenizer input exceeds 1 MiB"));
        }
        let mut tokens = Vec::new();
        if add_special
            && self.add_bos
            && let Some(bos) = self.bos
        {
            tokens.push(bos);
        }
        if !input.is_empty() {
            let fragments = self.partition_specials(input, parse_special);
            let mut is_first_normal = true;
            for fragment in fragments {
                match fragment {
                    Fragment::Special(id) => {
                        tokens.push(id);
                    }
                    Fragment::Raw(slice) => {
                        self.tokenize_fragment(slice, is_first_normal, &mut tokens)?;
                        is_first_normal = false;
                    }
                }
                if tokens.len() > MAX_TOKENS_OUTPUT {
                    return Err(invalid("tokenizer output exceeds bound"));
                }
            }
        }
        if add_special
            && self.add_sep
            && let Some(sep) = self.sep
        {
            tokens.push(sep);
        }
        if add_special
            && self.add_eos
            && let Some(eos) = self.eos
        {
            tokens.push(eos);
        }
        Ok(tokens)
    }

    fn partition_specials<'a>(&self, input: &'a [u8], parse_special: bool) -> Vec<Fragment<'a>> {
        let mut out = Vec::new();
        let mut cursor = 0_usize;
        let mut text_start = 0_usize;
        while cursor < input.len() {
            let rest = &input[cursor..];
            let matched = self.specials.iter().find(|(bytes, _, is_ctrl)| {
                (!*is_ctrl || parse_special) && rest.starts_with(bytes)
            });
            if let Some((bytes, id, _)) = matched {
                if cursor > text_start {
                    out.push(Fragment::Raw(&input[text_start..cursor]));
                }
                out.push(Fragment::Special(*id));
                cursor += bytes.len();
                text_start = cursor;
            } else {
                cursor += 1;
            }
        }
        if text_start < input.len() {
            out.push(Fragment::Raw(&input[text_start..]));
        }
        out
    }

    fn tokenize_fragment(
        &self,
        raw: &[u8],
        is_first_normal: bool,
        out: &mut Vec<i32>,
    ) -> Result<()> {
        match self.vocab_type {
            VocabType::None => {
                for _ in raw {
                    if let Some(unk) = self.unk {
                        out.push(unk);
                    }
                }
                Ok(())
            }
            VocabType::Spm => {
                self.tokenize_spm(raw, is_first_normal, out);
                Ok(())
            }
            VocabType::Bpe => self.tokenize_bpe(raw, out),
            VocabType::Wpm => self.tokenize_wpm(raw, out),
            VocabType::Ugm => {
                self.tokenize_ugm(raw, is_first_normal, out);
                Ok(())
            }
            VocabType::Rwkv | VocabType::Plamo2 => self.tokenize_greedy_bytes(raw, out),
        }
    }

    fn tokenize_spm(&self, raw: &[u8], is_first_normal: bool, out: &mut Vec<i32>) {
        let mut prepared = Vec::with_capacity(raw.len() + 8);
        if self.add_space_prefix && !self.treat_whitespace_as_suffix && is_first_normal {
            prepared.push(b' ');
        }
        prepared.extend_from_slice(raw);
        if self.add_space_prefix && self.treat_whitespace_as_suffix {
            prepared.push(b' ');
        }
        let escaped = if self.escape_whitespaces {
            replace_bytes(&prepared, b" ", "▁".as_bytes())
        } else {
            prepared
        };
        let chunks = split_utf8_chunks(&escaped);
        if chunks.is_empty() {
            return;
        }
        let n = chunks.len();
        let mut sym_bytes: Vec<Vec<u8>> = chunks;
        let mut prev: Vec<Option<usize>> = (0..n).map(|i| i.checked_sub(1)).collect();
        let mut next: Vec<Option<usize>> = (0..n)
            .map(|i| if i + 1 < n { Some(i + 1) } else { None })
            .collect();
        let mut alive = vec![true; n];
        let mut rev = vec![0_u64; n];
        let mut heap: SpmHeap = BinaryHeap::new();
        for left in 0..n.saturating_sub(1) {
            let right = left + 1;
            self.push_spm_bigram(&sym_bytes, left, right, &rev, &mut heap);
        }
        while let Some((_, Reverse(left), left_idx, right_idx, lrev, rrev)) = heap.pop() {
            if !alive[left_idx]
                || !alive[right_idx]
                || rev[left_idx] != lrev
                || rev[right_idx] != rrev
                || next[left_idx] != Some(right_idx)
            {
                continue;
            }
            let _ = left;
            let right_bytes = sym_bytes[right_idx].clone();
            sym_bytes[left_idx].extend_from_slice(&right_bytes);
            alive[right_idx] = false;
            rev[left_idx] += 1;
            let after = next[right_idx];
            next[left_idx] = after;
            if let Some(after_idx) = after {
                prev[after_idx] = Some(left_idx);
                self.push_spm_bigram(&sym_bytes, left_idx, after_idx, &rev, &mut heap);
            }
            if let Some(before_idx) = prev[left_idx] {
                self.push_spm_bigram(&sym_bytes, before_idx, left_idx, &rev, &mut heap);
            }
        }
        let mut cursor = Some(0_usize);
        while let Some(idx) = cursor {
            if alive[idx] {
                self.emit_spm_symbol(&sym_bytes[idx], out);
            }
            cursor = next[idx];
        }
    }

    fn push_spm_bigram(
        &self,
        sym_bytes: &[Vec<u8>],
        left: usize,
        right: usize,
        rev: &[u64],
        heap: &mut SpmHeap,
    ) {
        let mut combined = Vec::with_capacity(sym_bytes[left].len() + sym_bytes[right].len());
        combined.extend_from_slice(&sym_bytes[left]);
        combined.extend_from_slice(&sym_bytes[right]);
        if let Some(&id) = self.token_to_id.get(&combined)
            && let Ok(u) = usize::try_from(id)
        {
            heap.push((
                ScoreKey(self.entries[u].score),
                Reverse(left),
                left,
                right,
                rev[left],
                rev[right],
            ));
        }
    }

    fn emit_spm_symbol(&self, bytes: &[u8], out: &mut Vec<i32>) {
        if let Some(&id) = self.token_to_id.get(bytes) {
            out.push(id);
            return;
        }
        let mut fallback_emitted = false;
        for &byte in bytes {
            if let Some(id) = self.byte_tokens[usize::from(byte)] {
                out.push(id);
                fallback_emitted = true;
            }
        }
        if !fallback_emitted && let Some(unk) = self.unk {
            out.push(unk);
        }
    }

    fn encode_gpt_bytes(&self, raw: &[u8]) -> Vec<u8> {
        let mut encoded = Vec::with_capacity(raw.len() * 2);
        for &byte in raw {
            encoded.extend_from_slice(self.gpt_alphabet[usize::from(byte)].as_bytes());
        }
        encoded
    }

    fn tokenize_bpe(&self, raw: &[u8], out: &mut Vec<i32>) -> Result<()> {
        if self.ignore_merges && !raw.is_empty() {
            let whole = self.encode_gpt_bytes(raw);
            if let Some(&id) = self.token_to_id.get(&whole) {
                out.push(id);
                return Ok(());
            }
        }
        let Ok(text) = std::str::from_utf8(raw) else {
            // Invalid UTF-8 byte sequence: run BPE directly on byte-encoded slice.
            self.tokenize_bpe_piece(raw, out);
            return Ok(());
        };
        let pieces = split_text(text, &self.pre_patterns)?;
        for piece in pieces {
            self.tokenize_bpe_piece(piece.as_bytes(), out);
        }
        Ok(())
    }

    fn tokenize_bpe_piece(&self, raw: &[u8], out: &mut Vec<i32>) {
        if raw.is_empty() {
            return;
        }
        let n = raw.len();
        let mut sym_bytes: Vec<Vec<u8>> = raw
            .iter()
            .map(|&b| self.gpt_alphabet[usize::from(b)].as_bytes().to_vec())
            .collect();
        let mut prev: Vec<Option<usize>> = (0..n).map(|i| i.checked_sub(1)).collect();
        let mut next: Vec<Option<usize>> = (0..n)
            .map(|i| if i + 1 < n { Some(i + 1) } else { None })
            .collect();
        let mut alive = vec![true; n];
        let mut rev = vec![0_u64; n];
        let mut heap: BpeHeap = BinaryHeap::new();
        for left in 0..n.saturating_sub(1) {
            let right = left + 1;
            self.push_bpe_pair(&sym_bytes, left, right, &rev, &mut heap);
        }
        while let Some(Reverse((_, left_idx, right_idx, _, lrev, rrev))) = heap.pop() {
            if !alive[left_idx]
                || !alive[right_idx]
                || rev[left_idx] != lrev
                || rev[right_idx] != rrev
                || next[left_idx] != Some(right_idx)
            {
                continue;
            }
            let right_bytes = sym_bytes[right_idx].clone();
            sym_bytes[left_idx].extend_from_slice(&right_bytes);
            alive[right_idx] = false;
            rev[left_idx] += 1;
            let after = next[right_idx];
            next[left_idx] = after;
            if let Some(after_idx) = after {
                prev[after_idx] = Some(left_idx);
                self.push_bpe_pair(&sym_bytes, left_idx, after_idx, &rev, &mut heap);
            }
            if let Some(before_idx) = prev[left_idx] {
                self.push_bpe_pair(&sym_bytes, before_idx, left_idx, &rev, &mut heap);
            }
        }
        let mut cursor = Some(0_usize);
        while let Some(idx) = cursor {
            if alive[idx] {
                if let Some(&id) = self.token_to_id.get(&sym_bytes[idx]) {
                    out.push(id);
                } else {
                    for ch in String::from_utf8_lossy(&sym_bytes[idx]).chars() {
                        if let Some(raw_byte) = lookup_gpt_byte(&self.gpt_char_to_byte, ch) {
                            if let Some(id) = self.byte_tokens[usize::from(raw_byte)] {
                                out.push(id);
                            } else if let Some(unk) = self.unk {
                                out.push(unk);
                            }
                        } else if let Some(unk) = self.unk {
                            out.push(unk);
                        }
                    }
                }
            }
            cursor = next[idx];
        }
    }

    fn push_bpe_pair(
        &self,
        sym_bytes: &[Vec<u8>],
        left: usize,
        right: usize,
        rev: &[u64],
        heap: &mut BpeHeap,
    ) {
        if let Some(&rank) = self
            .merges
            .get(&(sym_bytes[left].clone(), sym_bytes[right].clone()))
        {
            heap.push(Reverse((rank, left, right, 0, rev[left], rev[right])));
        }
    }

    fn tokenize_wpm(&self, raw: &[u8], out: &mut Vec<i32>) -> Result<()> {
        let text = std::str::from_utf8(raw)
            .map_err(|_| invalid("WPM tokenization requires valid UTF-8 input"))?;
        let normalized = if self.lowercase_normalizer {
            text.to_lowercase()
        } else {
            text.to_owned()
        };
        let normalized = if self.strip_accents_normalizer {
            normalized
                .chars()
                .filter(|ch| !('\u{0300}'..='\u{036f}').contains(ch))
                .collect()
        } else {
            normalized
        };
        // Split on whitespace, CJK characters, and punctuation as WordPiece pre-tokenization.
        let mut words: Vec<String> = Vec::new();
        let mut current = String::new();
        for ch in normalized.chars() {
            if ch.is_whitespace() || is_control_char(ch) {
                if !current.is_empty() {
                    words.push(std::mem::take(&mut current));
                }
            } else if is_chinese_char(ch) || ch.is_ascii_punctuation() || is_unicode_punct(ch) {
                if !current.is_empty() {
                    words.push(std::mem::take(&mut current));
                }
                words.push(ch.to_string());
            } else {
                current.push(ch);
            }
        }
        if !current.is_empty() {
            words.push(current);
        }
        for word in words {
            if word.len() > 200 {
                if let Some(unk) = self.unk {
                    out.push(unk);
                }
                continue;
            }
            let prefixed = format!("▁{word}");
            let bytes = prefixed.as_bytes();
            let mut pos = 0_usize;
            let mut word_tokens = Vec::new();
            let mut ok = true;
            while pos < bytes.len() {
                let mut found = None;
                let mut end = bytes.len();
                while end > pos {
                    if prefixed.is_char_boundary(end)
                        && let Some(&id) = self.token_to_id.get(&bytes[pos..end])
                    {
                        found = Some((id, end));
                        break;
                    }
                    end -= 1;
                }
                if let Some((id, next_pos)) = found {
                    word_tokens.push(id);
                    pos = next_pos;
                } else {
                    ok = false;
                    break;
                }
            }
            if ok {
                out.extend_from_slice(&word_tokens);
            } else if let Some(unk) = self.unk {
                out.push(unk);
            }
        }
        Ok(())
    }

    fn tokenize_ugm(&self, raw: &[u8], is_first_normal: bool, out: &mut Vec<i32>) {
        let mut prepared = Vec::with_capacity(raw.len() + 4);
        if self.add_space_prefix && !self.treat_whitespace_as_suffix && is_first_normal {
            prepared.push(b' ');
        }
        prepared.extend_from_slice(raw);
        let escaped = if self.escape_whitespaces {
            replace_bytes(&prepared, b" ", "▁".as_bytes())
        } else {
            prepared
        };
        let n = escaped.len();
        if n == 0 {
            return;
        }
        // Forward Viterbi DP over byte positions using token log-scores.
        let mut best_score = vec![f64::NEG_INFINITY; n + 1];
        let mut back_len = vec![0_usize; n + 1];
        let mut back_id = vec![None::<i32>; n + 1];
        best_score[0] = 0.0;
        let max_token_len = self
            .entries
            .iter()
            .map(|e| e.raw.len())
            .max()
            .unwrap_or(1)
            .min(128);
        for i in 0..n {
            if !best_score[i].is_finite() {
                continue;
            }
            let max_end = (i + max_token_len).min(n);
            let mut matched_any = false;
            for j in (i + 1)..=max_end {
                if let Some(&id) = self.token_to_id.get(&escaped[i..j])
                    && let Ok(u) = usize::try_from(id)
                {
                    matched_any = true;
                    let candidate = best_score[i] + f64::from(self.entries[u].score);
                    if candidate > best_score[j] {
                        best_score[j] = candidate;
                        back_len[j] = j - i;
                        back_id[j] = Some(id);
                    }
                }
            }
            if !matched_any {
                // Unknown UTF-8 character or single byte fallback.
                let step = utf8_char_step(&escaped[i..]);
                let candidate = best_score[i] - 10.0;
                if candidate > best_score[i + step] {
                    best_score[i + step] = candidate;
                    back_len[i + step] = step;
                    back_id[i + step] = None;
                }
            }
        }
        let mut spans = Vec::new();
        let mut pos = n;
        while pos > 0 {
            let step = back_len[pos].max(1);
            let start = pos - step;
            spans.push((start, pos, back_id[pos]));
            pos = start;
        }
        spans.reverse();
        for (start, end, opt_id) in spans {
            if let Some(id) = opt_id {
                out.push(id);
            } else {
                self.emit_spm_symbol(&escaped[start..end], out);
            }
        }
    }

    fn tokenize_greedy_bytes(&self, raw: &[u8], out: &mut Vec<i32>) -> Result<()> {
        let max_len = self
            .entries
            .iter()
            .filter(|e| matches!(e.kind, TokenType::Normal | TokenType::UserDefined))
            .map(|e| e.raw.len())
            .max()
            .unwrap_or(1)
            .min(128);
        let mut pos = 0_usize;
        while pos < raw.len() {
            let mut end = (pos + max_len).min(raw.len());
            let mut matched = None;
            while end > pos {
                if let Some(&id) = self.token_to_id.get(&raw[pos..end])
                    && let Ok(u) = usize::try_from(id)
                    && matches!(
                        self.entries[u].kind,
                        TokenType::Normal | TokenType::UserDefined
                    )
                {
                    matched = Some((id, end));
                    break;
                }
                end -= 1;
            }
            if let Some((id, next_pos)) = matched {
                out.push(id);
                pos = next_pos;
            } else if let Some(id) = self.byte_tokens[usize::from(raw[pos])] {
                out.push(id);
                pos += 1;
            } else if let Some(unk) = self.unk {
                out.push(unk);
                pos += 1;
            } else {
                return Err(invalid("byte missing from greedy tokenizer vocabulary"));
            }
        }
        Ok(())
    }
}

enum Fragment<'a> {
    Special(i32),
    Raw(&'a [u8]),
}

#[derive(Clone, Copy, PartialEq)]
struct ScoreKey(f32);

impl Eq for ScoreKey {}

impl PartialOrd for ScoreKey {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ScoreKey {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

fn replace_bytes(input: &[u8], pattern: &[u8], replacement: &[u8]) -> Vec<u8> {
    if pattern.is_empty() {
        return input.to_vec();
    }
    let mut out = Vec::with_capacity(input.len());
    let mut i = 0_usize;
    while i < input.len() {
        if input[i..].starts_with(pattern) {
            out.extend_from_slice(replacement);
            i += pattern.len();
        } else {
            out.push(input[i]);
            i += 1;
        }
    }
    out
}

fn utf8_char_step(bytes: &[u8]) -> usize {
    let Some(&b0) = bytes.first() else {
        return 0;
    };
    let expected = match b0 {
        0x00..=0x7f => 1,
        0xc2..=0xdf => 2,
        0xe0..=0xef => 3,
        0xf0..=0xf4 => 4,
        _ => return 1,
    };
    if bytes.len() >= expected && std::str::from_utf8(&bytes[..expected]).is_ok() {
        expected
    } else {
        1
    }
}

fn split_utf8_chunks(bytes: &[u8]) -> Vec<Vec<u8>> {
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0_usize;
    while i < bytes.len() {
        let step = utf8_char_step(&bytes[i..]);
        out.push(bytes[i..i + step].to_vec());
        i += step;
    }
    out
}

const fn is_control_char(ch: char) -> bool {
    !matches!(ch, '\t' | '\n' | '\r') && ch.is_control()
}

fn is_chinese_char(ch: char) -> bool {
    let cp = u32::from(ch);
    matches!(
        cp,
        0x4e00..=0x9fff
            | 0x3400..=0x4dbf
            | 0x20000..=0x2a6df
            | 0x2a700..=0x2b73f
            | 0x2b740..=0x2b81f
            | 0x2b820..=0x2ceaf
            | 0xf900..=0xfaff
            | 0x2f800..=0x2fa1f
    )
}

fn is_unicode_punct(ch: char) -> bool {
    let cp = u32::from(ch);
    matches!(cp, 33..=47 | 58..=64 | 91..=96 | 123..=126)
}
