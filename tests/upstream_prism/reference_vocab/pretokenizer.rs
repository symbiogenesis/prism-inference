//! Clean-room pre-tokenizer regex tables and GPT-2 byte-alphabet mapping.
//!
//! No `bonsai_prism_runtime` or `tokenizers` crate is called here.

use fancy_regex::Regex;

use super::super::{Result, invalid};

const MAX_SPLIT_PIECES: usize = 262_144;

const fn is_direct_gpt_byte(byte: u8) -> bool {
    matches!(byte, b'!'..=b'~' | 0xa1..=0xac | 0xae..=u8::MAX)
}

pub(super) fn build_gpt_byte_alphabet() -> Result<[String; 256]> {
    let mut next_extra = 256_u32;
    let mut out: [String; 256] = std::array::from_fn(|_| String::new());
    for byte in 0..=u8::MAX {
        let scalar = if is_direct_gpt_byte(byte) {
            u32::from(byte)
        } else {
            let value = next_extra;
            next_extra += 1;
            value
        };
        let ch =
            char::from_u32(scalar).ok_or_else(|| invalid("invalid GPT-2 byte alphabet scalar"))?;
        out[usize::from(byte)] = ch.to_string();
    }
    Ok(out)
}

pub(super) fn build_gpt_char_to_byte(alphabet: &[String; 256]) -> Vec<(char, u8)> {
    let mut pairs = Vec::with_capacity(256);
    for (byte, encoded) in alphabet.iter().enumerate() {
        if let Some(ch) = encoded.chars().next()
            && let Ok(raw) = u8::try_from(byte)
        {
            pairs.push((ch, raw));
        }
    }
    pairs.sort_unstable_by_key(|(ch, _)| *ch);
    pairs
}

pub(super) fn lookup_gpt_byte(table: &[(char, u8)], ch: char) -> Option<u8> {
    table
        .binary_search_by_key(&ch, |(entry, _)| *entry)
        .ok()
        .map(|idx| table[idx].1)
}

pub(super) fn bpe_pretokenizer_patterns(pre: &str) -> &'static [&'static str] {
    match pre {
        "llama3" | "llama-v3" | "llama-bpe" | "falcon3" | "tiny_aya" | "smollm" | "chameleon" => &[
            r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
        ],
        "llama4" | "tekken" | "pixtral" => &[
            r"[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}{1,3}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
        ],
        "qwen2" | "command-r" | "deepseek-r1-qwen" | "cohere2moe" => &[
            r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
        ],
        "qwen35" | "hunyuan" | "hunyuan-dense" | "grok-2" => &[
            r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
        ],
        "deepseek-llm" => &[
            "[\r\n]",
            r"\s{2,}",
            r"[']s|[']t|[']re|[']ve|[']m|[']ll|[']d| ?\p{L}+| ?\p{N}| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+",
        ],
        "deepseek-coder" => &[
            "[\r\n]",
            r"\s?\p{L}+",
            r"\p{N}",
            r"[\x{4e00}-\x{9fa5}]+",
            r"[^\s\p{L}\p{N}]+",
            r"\s+",
        ],
        "deepseek-v3" => &[
            r"\p{N}{1,3}",
            r"[\x{4e00}-\x{9fa5}\x{3040}-\x{309f}\x{30a0}-\x{30ff}]+",
            r#"[!#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~"\\]+"#,
            r"\s+",
            r"[A-Za-z]+",
            r"[^\s\p{L}\p{N}]+",
        ],
        "falcon" => &[
            r"[\p{N}]{1,3}",
            r" ?\p{L}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+",
        ],
        "starcoder" | "gpt-neox" | "refact" | "codeshell" => &[
            r"\p{N}",
            r"'(?:s|t|re|ve|m|ll|d)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+",
        ],
        "gpt-4o" => &[
            r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+",
        ],
        "superbpe" => &[
            r"\p{N}+",
            r"(?=(\s))\s+(?!\S)|\s+",
            r"'(?:s|t|re|ve|m|ll|d)| ?\p{L}+| ?[^\s\p{L}\p{N}]+",
        ],
        _ => &[r"'(?:s|t|re|ve|m|ll|d)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"],
    }
}

pub(super) fn compile_pretokenizer(pre: &str) -> Result<Vec<Regex>> {
    bpe_pretokenizer_patterns(pre)
        .iter()
        .map(|pattern| {
            Regex::new(pattern)
                .map_err(|error| invalid(format!("invalid BPE pre-tokenizer regex: {error}")))
        })
        .collect()
}

pub(super) fn split_text<'a>(text: &'a str, patterns: &[Regex]) -> Result<Vec<&'a str>> {
    if text.is_empty() {
        return Ok(Vec::new());
    }
    let mut pieces = vec![text];
    for regex in patterns {
        let mut next = Vec::with_capacity(pieces.len());
        for piece in pieces {
            let mut end = 0_usize;
            for found in regex.find_iter(piece) {
                let found = found
                    .map_err(|error| invalid(format!("BPE pretokenizer regex failed: {error}")))?;
                if found.start() > end {
                    next.push(&piece[end..found.start()]);
                }
                if found.end() > found.start() {
                    next.push(found.as_str());
                }
                end = found.end();
            }
            if end < piece.len() {
                next.push(&piece[end..]);
            }
            if next.len() > MAX_SPLIT_PIECES {
                return Err(invalid("vocabulary pretokenizer piece bound exceeded"));
            }
        }
        pieces = next;
    }
    Ok(pieces)
}
