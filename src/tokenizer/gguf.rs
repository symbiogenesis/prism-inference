//! Safe, bounded tokenization for the tokenizer families declared by GGUF.
//!
//! Algorithms follow the pinned Prism vocabulary contract. Metadata is used
//! directly: no publisher tokenizer files or reference token observations are
//! needed at runtime. The specialized Qwen35 API remains independently strict.
//! Algorithm port: PrismML-Eng/llama.cpp `73bfe6cc485554822f80c473a3c0c4f25e80f41b`,
//! `src/llama-vocab.cpp`; upstream MIT attribution is retained in
//! `tools/prism/upstream/LICENSE`.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

use aho_corasick::{AhoCorasick, AhoCorasickKind, MatchKind};
use bonsai_gguf::{Metadata, MetadataArray, MetadataValue};
use fancy_regex::Regex;
use unicode_normalization_alignments::UnicodeNormalization;

#[path = "gguf_charsmap.rs"]
mod charsmap;
use charsmap::CharsMap;

use crate::pretokenizer::Qwen35PreTokenizer;
use crate::vocabulary::{
    MAX_PIECE_BYTES, MAX_VOCABULARY_BYTES, MAX_VOCABULARY_TOKENS, decode_character,
};
use crate::{Result, invalid};

const LIMIT: usize = 1024 * 1024;

// Search all eligible special spellings together instead of scanning the full
// remaining input once per spelling. IDs stay ordered so duplicate spellings
// retain the source's lowest-ID tie. Very large tables retain the linear path
// rather than adding unbounded automaton construction to model loading.
#[derive(Debug)]
struct SpecialIndex {
    ids: Vec<usize>,
    matcher: Option<AhoCorasick>,
}

impl SpecialIndex {
    fn new(tokens: &[String], types: &[i32], parse_special: bool) -> Self {
        let ids: Vec<_> = types
            .iter()
            .enumerate()
            .filter_map(|(id, &kind)| {
                (!tokens[id].is_empty() && (kind == 4 || (parse_special && matches!(kind, 2 | 3))))
                    .then_some(id)
            })
            .collect();
        let matcher =
            if ids.is_empty() || ids.iter().map(|&id| tokens[id].len()).sum::<usize>() > LIMIT {
                None
            } else {
                AhoCorasick::builder()
                    .kind(Some(AhoCorasickKind::ContiguousNFA))
                    .match_kind(MatchKind::LeftmostLongest)
                    .build(ids.iter().map(|&id| tokens[id].as_bytes()))
                    .ok()
            };
        Self { ids, matcher }
    }

    fn find(&self, tokens: &[String], input: &[u8]) -> Option<(usize, usize)> {
        if let Some(matcher) = &self.matcher {
            return matcher
                .find(input)
                .map(|found| (found.start(), self.ids[found.pattern().as_usize()]));
        }
        self.ids
            .iter()
            .filter_map(|&id| {
                input
                    .windows(tokens[id].len())
                    .position(|part| part == tokens[id].as_bytes())
                    .map(|position| (position, std::cmp::Reverse(tokens[id].len()), id))
            })
            .min()
            .map(|(position, _, id)| (position, id))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Family {
    Bpe,
    Spm,
    WordPiece,
    Unigram,
    Rwkv,
    Plamo,
}

#[derive(Debug)]
struct Symbol {
    start: usize,
    end: usize,
    previous: Option<usize>,
    next: Option<usize>,
}

#[derive(Debug)]
struct Pair {
    priority: f64,
    left: usize,
    right: usize,
    size: usize,
}

// A lexicographic index narrows the vocabulary range one input byte at a time.
// It retains one index per nonempty token, without a heap node per vocabulary
// byte. Matches arrive shortest first, then in original ID order for duplicates.
#[derive(Debug)]
struct PrefixIndex(Vec<usize>);

impl PrefixIndex {
    fn new(raw: &[Vec<u8>]) -> Self {
        let mut entries: Vec<_> = (0..raw.len()).filter(|&id| !raw[id].is_empty()).collect();
        entries.sort_unstable_by(|&left, &right| {
            raw[left].cmp(&raw[right]).then_with(|| left.cmp(&right))
        });
        Self(entries)
    }

    fn matching<'a>(&'a self, raw: &'a [Vec<u8>], input: &'a [u8]) -> PrefixMatches<'a> {
        PrefixMatches {
            entries: &self.0,
            raw,
            input,
            depth: 0,
        }
    }
}

struct PrefixMatches<'a> {
    entries: &'a [usize],
    raw: &'a [Vec<u8>],
    input: &'a [u8],
    depth: usize,
}

impl Iterator for PrefixMatches<'_> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let &first = self.entries.first()?;
            if self.raw[first].len() == self.depth {
                self.entries = &self.entries[1..];
                return Some(first);
            }
            let &byte = self.input.get(self.depth)?;
            let start = self
                .entries
                .partition_point(|&id| self.raw[id][self.depth] < byte);
            let end = self
                .entries
                .partition_point(|&id| self.raw[id][self.depth] <= byte);
            self.entries = &self.entries[start..end];
            self.depth += 1;
        }
    }
}
impl PartialEq for Pair {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Pair {}
impl PartialOrd for Pair {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Pair {
    fn cmp(&self, other: &Self) -> Ordering {
        self.priority
            .total_cmp(&other.priority)
            .then_with(|| other.left.cmp(&self.left))
    }
}

#[derive(Default)]
struct MergeScratch {
    nodes: Vec<Symbol>,
    queue: BinaryHeap<Pair>,
    rank_key: String,
}

/// Immutable production tokenizer loaded solely from bounded GGUF metadata.
#[derive(Debug)]
#[allow(
    clippy::struct_excessive_bools,
    reason = "GGUF independently declares these tokenizer flags"
)]
pub struct GgufTokenizer {
    family: Family,
    model: String,
    pre: String,
    tokens: Vec<String>,
    types: Vec<i32>,
    specials: SpecialIndex,
    user_defined: SpecialIndex,
    scores: Vec<f32>,
    plamo_scores: Vec<Option<i32>>,
    lookup: HashMap<String, u32>,
    ranks: HashMap<String, usize>,
    prefixes: PrefixIndex,
    raw: Vec<Vec<u8>>,
    byte_alphabet: [char; 256],
    byte_tokens: [Option<u32>; 256],
    minimum_score: f32,
    regexes: Vec<Regex>,
    charsmap: Option<CharsMap>,
    bos: Option<u32>,
    eos: Option<u32>,
    unknown: Option<u32>,
    separator: Option<u32>,
    add_bos: bool,
    add_eos: bool,
    prefix_space: bool,
    remove_spaces: bool,
    escape_spaces: bool,
    lowercase: bool,
    strip_accents: bool,
    ignore_merges: bool,
}

fn text<'a>(metadata: &'a Metadata, name: &str) -> Option<&'a str> {
    metadata
        .get(&format!("tokenizer.ggml.{name}"))
        .and_then(MetadataValue::as_str)
}
fn flag(metadata: &Metadata, name: &str, default: bool) -> Result<bool> {
    metadata
        .get(&format!("tokenizer.ggml.{name}"))
        .map_or(Ok(default), |value| {
            value
                .as_bool()
                .ok_or_else(|| invalid(format!("tokenizer flag {name} is not BOOL")))
        })
}
fn id(metadata: &Metadata, name: &str, default: Option<u32>, count: usize) -> Result<Option<u32>> {
    let value = metadata
        .get(&format!("tokenizer.ggml.{name}_token_id"))
        .map_or(Ok(default), |value| {
            if let MetadataValue::U32(value) = value {
                Ok(Some(*value))
            } else {
                Err(invalid("tokenizer ID is not UINT32"))
            }
        })?;
    Ok(value.filter(|value| usize::try_from(*value).is_ok_and(|value| value < count)))
}

impl GgufTokenizer {
    /// Validates and owns a tokenizer without requiring model weights.
    ///
    /// # Errors
    /// Rejects unsupported families, malformed metadata, inconsistent counts,
    /// nonfinite scores, and vocabulary or merge resource limits.
    #[allow(
        clippy::too_many_lines,
        reason = "Validate all bounded GGUF arrays before constructing the tokenizer"
    )]
    pub fn from_metadata(metadata: &Metadata, count: usize) -> Result<Self> {
        let model = text(metadata, "model").ok_or_else(|| invalid("tokenizer model missing"))?;
        let family = match model {
            "gpt2" | "hybriddna" | "gemma4" | "whitespace" => Family::Bpe,
            "llama" => Family::Spm,
            "bert" => Family::WordPiece,
            "t5" => Family::Unigram,
            "rwkv" => Family::Rwkv,
            "plamo2" => Family::Plamo,
            _ => return Err(invalid(format!("unsupported tokenizer model {model}"))),
        };
        let pre = if model == "gemma4" {
            "gemma4"
        } else {
            text(metadata, "pre").unwrap_or("default")
        };
        if count == 0 || count > MAX_VOCABULARY_TOKENS {
            return Err(invalid("tokenizer vocabulary count exceeds bound"));
        }
        let Some(MetadataArray::String(tokens)) = metadata
            .get("tokenizer.ggml.tokens")
            .and_then(MetadataValue::as_array)
        else {
            return Err(invalid("tokenizer tokens require STRING array"));
        };
        if tokens.len() != count
            || tokens.iter().any(|token| token.len() > MAX_PIECE_BYTES)
            || tokens.iter().map(String::len).sum::<usize>() > MAX_VOCABULARY_BYTES
        {
            return Err(invalid("tokenizer vocabulary bytes/count exceed bound"));
        }
        let types = match metadata
            .get("tokenizer.ggml.token_type")
            .and_then(MetadataValue::as_array)
        {
            Some(MetadataArray::I32(types))
                if types.len() == count && types.iter().all(|kind| (1..=6).contains(kind)) =>
            {
                types.clone()
            }
            None => vec![1; count],
            _ => return Err(invalid("tokenizer types require matching INT32 array")),
        };
        let scores = match metadata
            .get("tokenizer.ggml.scores")
            .and_then(MetadataValue::as_array)
        {
            Some(MetadataArray::F32(scores))
                if scores.len() >= count && scores.iter().all(|v| v.is_finite()) =>
            {
                scores[..count].to_vec()
            }
            None => vec![0.0; count],
            _ => return Err(invalid("tokenizer scores require finite FLOAT32 array")),
        };
        let mut lookup = HashMap::new();
        lookup.try_reserve(count)?;
        let mut raw = Vec::new();
        let indexed = matches!(
            family,
            Family::WordPiece | Family::Unigram | Family::Rwkv | Family::Plamo
        );
        if indexed {
            raw.try_reserve_exact(count)?;
        }
        for (index, token) in tokens.iter().enumerate() {
            let token_id = u32::try_from(index)?;
            lookup.insert(token.clone(), token_id);
            if indexed {
                raw.push(if family == Family::Rwkv {
                    unescape_rwkv(token)?
                } else {
                    token.as_bytes().to_vec()
                });
            }
        }
        let prefixes = PrefixIndex::new(&raw);
        let byte_tokens = std::array::from_fn(|value| {
            u8::try_from(value).ok().and_then(|value| {
                lookup
                    .get(&format!("<0x{value:02X}>"))
                    .or_else(|| lookup.get(&char::from(value).to_string()))
                    .copied()
            })
        });
        let minimum_score = scores
            .iter()
            .zip(&types)
            .filter(|(_, kind)| **kind == 1)
            .map(|(score, _)| *score)
            .reduce(f32::min)
            .unwrap_or(0.0);
        // Failed conversions retain the original lazy error: a vocabulary may
        // contain an out-of-range score on a token that this request never uses.
        let plamo_scores = if family == Family::Plamo {
            scores
                .iter()
                .map(|&score| rounded_score(score).ok())
                .collect()
        } else {
            Vec::new()
        };
        let mut ranks = HashMap::new();
        if family == Family::Bpe {
            let Some(MetadataArray::String(merges)) = metadata
                .get("tokenizer.ggml.merges")
                .and_then(MetadataValue::as_array)
            else {
                return Err(invalid("BPE requires STRING merges"));
            };
            if merges.len() > 1_000_000
                || merges.iter().map(String::len).sum::<usize>() > 64 * LIMIT
            {
                return Err(invalid("BPE merge metadata exceeds bound"));
            }
            ranks.try_reserve(merges.len())?;
            for (rank, merge) in merges.iter().enumerate() {
                ranks.entry(merge.clone()).or_insert(rank);
            }
        }
        let regexes = match family {
            Family::Bpe => patterns(pre)?,
            Family::WordPiece => vec![r"\p{P}|[!-/:-@\[-`{-~]|[^\s\p{P}!-/:-@\[-`{-~]+"],
            _ => Vec::new(),
        }
        .iter()
        .map(|pattern| Regex::new(pattern).map_err(Into::into))
        .collect::<Result<Vec<_>>>()?;
        let charsmap = match metadata
            .get("tokenizer.ggml.precompiled_charsmap")
            .and_then(MetadataValue::as_array)
        {
            Some(MetadataArray::U8(bytes)) => Some(CharsMap::from_bytes(bytes)?),
            Some(MetadataArray::I8(bytes)) => Some(CharsMap::from_bytes(
                &bytes.iter().map(|v| v.to_ne_bytes()[0]).collect::<Vec<_>>(),
            )?),
            None => None,
            _ => return Err(invalid("tokenizer charsmap requires byte array")),
        };
        let (bos, eos, unk, sep) = match family {
            Family::Spm | Family::Plamo => (Some(1), Some(2), Some(0), None),
            Family::WordPiece => (Some(101), None, Some(100), Some(102)),
            Family::Unigram => (None, Some(1), Some(2), None),
            Family::Bpe => (Some(11), Some(11), None, None),
            Family::Rwkv => (None, None, None, None),
        };
        let lowercase = flag(
            metadata,
            "normalizer.lowercase",
            family == Family::WordPiece,
        )?;
        let specials = SpecialIndex::new(tokens, &types, true);
        let user_defined = SpecialIndex::new(tokens, &types, false);
        Ok(Self {
            family,
            model: model.to_owned(),
            pre: pre.to_owned(),
            tokens: tokens.clone(),
            types,
            specials,
            user_defined,
            scores,
            plamo_scores,
            lookup,
            ranks,
            prefixes,
            raw,
            byte_alphabet: byte_alphabet()?,
            byte_tokens,
            minimum_score,
            regexes,
            charsmap,
            bos: id(metadata, "bos", bos, count)?,
            eos: id(metadata, "eos", eos, count)?,
            unknown: id(metadata, "unknown", unk, count)?,
            separator: id(metadata, "seperator", sep, count)?,
            add_bos: flag(
                metadata,
                "add_bos_token",
                family == Family::Spm || matches!(pre, "llama3" | "llama-v3" | "llama-bpe"),
            )?,
            add_eos: flag(metadata, "add_eos_token", family == Family::Unigram)?,
            prefix_space: flag(
                metadata,
                "add_space_prefix",
                family == Family::Spm || family == Family::Unigram,
            )?,
            remove_spaces: flag(metadata, "remove_extra_whitespaces", false)?,
            escape_spaces: family == Family::Spm || family == Family::Unigram || pre == "gemma4",
            lowercase,
            strip_accents: flag(metadata, "normalizer.strip_accents", lowercase)?,
            ignore_merges: matches!(pre, "llama3" | "llama-v3" | "llama-bpe"),
        })
    }

    /// Number of validated token IDs.
    #[must_use]
    pub const fn vocab_size(&self) -> usize {
        self.tokens.len()
    }

    /// Pinned GGUF vocabulary family discriminator (SPM=1 through PLaMo=6).
    #[must_use]
    pub const fn vocabulary_type(&self) -> u32 {
        match self.family {
            Family::Spm => 1,
            Family::Bpe => 2,
            Family::WordPiece => 3,
            Family::Unigram => 4,
            Family::Rwkv => 5,
            Family::Plamo => 6,
        }
    }

    /// Validated beginning-of-sequence ID, including family defaults.
    #[must_use]
    pub const fn bos_token(&self) -> Option<u32> {
        self.bos
    }

    /// Encodes a UTF-8 request, with explicit source special-token policies.
    ///
    /// User-defined tokens are always recognized; control tokens are recognized
    /// only with `parse_special`. Insertion uses family defaults overridden by
    /// the immutable metadata. No chat template is applied.
    ///
    /// # Errors
    /// Rejects malformed UTF-8, input/output limits, unavailable byte fallback,
    /// and missing enabled boundary IDs.
    pub fn encode(&self, bytes: &[u8], add_special: bool, parse_special: bool) -> Result<Vec<u32>> {
        if bytes.len() > LIMIT {
            return Err(invalid("tokenizer input exceeds byte limit"));
        }
        let text = match std::str::from_utf8(bytes) {
            Ok(text) => text,
            Err(_) if self.family == Family::Spm => {
                return self.encode_spm_bytes(bytes, add_special, parse_special);
            }
            Err(error) => return Err(error.into()),
        };
        let mut output = Vec::new();
        if add_special && (self.add_bos || self.family == Family::WordPiece) {
            output.push(
                self.bos
                    .ok_or_else(|| invalid("tokenizer BOS unavailable"))?,
            );
        }
        let mut offset = 0;
        while offset < text.len() {
            let remaining = &text[offset..];
            let special = self
                .special_index(parse_special)
                .find(&self.tokens, remaining.as_bytes());
            let end = special.map_or(text.len(), |(position, _)| offset + position);
            if end > offset {
                self.ordinary(&text[offset..end], true, &mut output)?;
            }
            if let Some((_, index)) = special {
                output.push(u32::try_from(index)?);
                offset = end + self.tokens[index].len();
            } else {
                break;
            }
            if output.len() > 262_144 {
                return Err(invalid("tokenizer output exceeds token bound"));
            }
        }
        if add_special {
            if self.family == Family::WordPiece {
                output.push(
                    self.separator
                        .ok_or_else(|| invalid("tokenizer separator unavailable"))?,
                );
            } else if self.add_eos {
                output.push(
                    self.eos
                        .ok_or_else(|| invalid("tokenizer EOS unavailable"))?,
                );
            }
        }
        if output.len() > 262_144 {
            return Err(invalid("tokenizer output exceeds token bound"));
        }
        Ok(output)
    }

    fn encode_spm_bytes(
        &self,
        bytes: &[u8],
        add_special: bool,
        parse_special: bool,
    ) -> Result<Vec<u32>> {
        let mut output = Vec::new();
        if add_special && self.add_bos {
            output.push(self.bos.ok_or_else(|| invalid("SPM BOS unavailable"))?);
        }
        let mut offset = 0;
        while offset < bytes.len() {
            let special = self
                .special_index(parse_special)
                .find(&self.tokens, &bytes[offset..]);
            let end = special.map_or(bytes.len(), |(position, _)| offset + position);
            if end > offset {
                let mut escaped = Vec::new();
                if self.prefix_space {
                    escaped.extend_from_slice("▁".as_bytes());
                }
                for &byte in &bytes[offset..end] {
                    if byte == b' ' {
                        escaped.extend_from_slice("▁".as_bytes());
                    } else {
                        escaped.push(byte);
                    }
                }
                self.merge_spm_bytes(&escaped, &mut output)?;
            }
            if let Some((_, index)) = special {
                output.push(u32::try_from(index)?);
                offset = end + self.tokens[index].len();
            } else {
                break;
            }
        }
        if add_special && self.add_eos {
            output.push(self.eos.ok_or_else(|| invalid("SPM EOS unavailable"))?);
        }
        if output.len() > 262_144 {
            return Err(invalid("tokenizer output exceeds token bound"));
        }
        Ok(output)
    }

    const fn special_index(&self, parse_special: bool) -> &SpecialIndex {
        if parse_special {
            &self.specials
        } else {
            &self.user_defined
        }
    }

    fn merge_spm_bytes(&self, bytes: &[u8], output: &mut Vec<u32>) -> Result<()> {
        let mut nodes: Vec<Symbol> = Vec::new();
        let mut offset = 0;
        while offset < bytes.len() {
            let length = match bytes[offset] {
                0..=0xbf => 1,
                0xc0..=0xdf => 2,
                0xe0..=0xef => 3,
                _ => 4,
            };
            let index = nodes.len();
            let end = (offset + length).min(bytes.len());
            nodes.push(Symbol {
                start: offset,
                end,
                previous: index.checked_sub(1),
                next: (end < bytes.len()).then_some(index + 1),
            });
            offset = end;
        }
        let mut queue = BinaryHeap::new();
        let candidate =
            |nodes: &[Symbol], left: usize, queue: &mut BinaryHeap<Pair>| -> Result<()> {
                let Some(right) = nodes[left].next else {
                    return Ok(());
                };
                if let Ok(text) = std::str::from_utf8(&bytes[nodes[left].start..nodes[right].end])
                    && let Some(&id) = self.lookup.get(text)
                {
                    queue.push(Pair {
                        priority: f64::from(self.scores[usize::try_from(id)?]),
                        left,
                        right,
                        size: nodes[right].end - nodes[left].start,
                    });
                }
                Ok(())
            };
        for index in 0..nodes.len().saturating_sub(1) {
            candidate(&nodes, index, &mut queue)?;
        }
        while let Some(pair) = queue.pop() {
            if nodes[pair.left].start == nodes[pair.left].end
                || nodes[pair.right].start == nodes[pair.right].end
                || nodes[pair.left].next != Some(pair.right)
                || nodes[pair.right].end - nodes[pair.left].start != pair.size
            {
                continue;
            }
            let next = nodes[pair.right].next;
            nodes[pair.left].end = nodes[pair.right].end;
            nodes[pair.left].next = next;
            nodes[pair.right].end = nodes[pair.right].start;
            if let Some(next) = next {
                nodes[next].previous = Some(pair.left);
            }
            if let Some(previous) = nodes[pair.left].previous {
                candidate(&nodes, previous, &mut queue)?;
            }
            candidate(&nodes, pair.left, &mut queue)?;
        }
        let mut current = (!nodes.is_empty()).then_some(0);
        while let Some(index) = current {
            let piece = &bytes[nodes[index].start..nodes[index].end];
            if let Ok(text) = std::str::from_utf8(piece)
                && let Some(&id) = self.lookup.get(text)
            {
                output.push(id);
            } else {
                for &byte in piece {
                    output.push(self.byte(byte)?);
                }
            }
            current = nodes[index].next;
        }
        Ok(())
    }

    fn ordinary(&self, text: &str, previous_special: bool, output: &mut Vec<u32>) -> Result<()> {
        match self.family {
            Family::Bpe => self.bpe(text, output),
            Family::Spm => {
                let text = format!(
                    "{}{}",
                    if self.prefix_space && previous_special {
                        " "
                    } else {
                        ""
                    },
                    text
                )
                .replace(' ', "▁");
                self.merge(&text, false, output)
            }
            Family::WordPiece => self.wordpiece(text, output),
            Family::Unigram => self.unigram(text, output),
            Family::Rwkv => self.greedy(text.as_bytes(), output),
            Family::Plamo => self.plamo(text, output),
        }
    }

    fn byte(&self, value: u8) -> Result<u32> {
        self.byte_tokens[usize::from(value)]
            .ok_or_else(|| invalid(format!("tokenizer byte {value} unavailable")))
    }

    fn greedy(&self, bytes: &[u8], output: &mut Vec<u32>) -> Result<()> {
        let mut offset = 0;
        while offset < bytes.len() {
            let id = self
                .prefixes
                .matching(&self.raw, &bytes[offset..])
                .max_by_key(|&id| (self.raw[id].len(), std::cmp::Reverse(id)))
                .ok_or_else(|| invalid("tokenizer has no matching byte token"))?;
            offset += self.raw[id].len();
            output.push(u32::try_from(id)?);
        }
        Ok(())
    }

    fn bpe(&self, input: &str, output: &mut Vec<u32>) -> Result<()> {
        if self.model == "hybriddna" && input.contains("<dna>") {
            return self.dna(input, output);
        }
        let text = if self.escape_spaces {
            input.replace(' ', "▁")
        } else {
            input.to_owned()
        };
        let pieces = if self.pre == "qwen35" {
            let metadata = Metadata::from([
                ("tokenizer.ggml.model".into(), "gpt2".into()),
                ("tokenizer.ggml.pre".into(), "qwen35".into()),
            ]);
            Qwen35PreTokenizer::from_metadata(&metadata)?.split(&text)?
        } else {
            let mut pieces = vec![text.as_str()];
            for regex in &self.regexes {
                let mut split = Vec::new();
                for piece in pieces {
                    let mut start = 0;
                    for found in regex.find_iter(piece) {
                        let found = found?;
                        if found.start() > start {
                            split.push(&piece[start..found.start()]);
                        }
                        if found.end() > found.start() {
                            split.push(found.as_str());
                        }
                        start = found.end();
                    }
                    if start < piece.len() {
                        split.push(&piece[start..]);
                    }
                }
                pieces = split;
            }
            pieces
        };
        let mut scratch = MergeScratch::default();
        for piece in pieces {
            let encoded = if self.pre == "gemma4" || self.model == "whitespace" {
                piece.to_owned()
            } else {
                piece
                    .bytes()
                    .map(|byte| self.byte_alphabet[usize::from(byte)])
                    .collect()
            };
            if self.ignore_merges
                && let Some(&id) = self.lookup.get(&encoded)
            {
                output.push(id);
                continue;
            }
            self.merge_with_scratch(&encoded, true, output, &mut scratch)?;
        }
        Ok(())
    }

    fn merge(&self, text: &str, ranked: bool, output: &mut Vec<u32>) -> Result<()> {
        self.merge_with_scratch(text, ranked, output, &mut MergeScratch::default())
    }

    fn merge_with_scratch(
        &self,
        text: &str,
        ranked: bool,
        output: &mut Vec<u32>,
        scratch: &mut MergeScratch,
    ) -> Result<()> {
        let MergeScratch {
            nodes,
            queue,
            rank_key,
        } = scratch;
        nodes.clear();
        queue.clear();
        nodes.extend(
            text.char_indices()
                .enumerate()
                .map(|(index, (start, value))| Symbol {
                    start,
                    end: start + value.len_utf8(),
                    previous: index.checked_sub(1),
                    next: Some(index + 1),
                }),
        );
        let count = nodes.len();
        if count == 0 {
            return Ok(());
        }
        nodes[count - 1].next = None;
        for left in 0..count.saturating_sub(1) {
            self.candidate(text, nodes, left, ranked, queue, rank_key)?;
        }
        while let Some(pair) = queue.pop() {
            if nodes[pair.left].start == nodes[pair.left].end
                || nodes[pair.left].next != Some(pair.right)
                || nodes[pair.right].start == nodes[pair.right].end
                || nodes[pair.left].end - nodes[pair.left].start + nodes[pair.right].end
                    - nodes[pair.right].start
                    != pair.size
            {
                continue;
            }
            let next = nodes[pair.right].next;
            nodes[pair.left].end = nodes[pair.right].end;
            nodes[pair.left].next = next;
            nodes[pair.right].end = nodes[pair.right].start;
            if let Some(next) = next {
                nodes[next].previous = Some(pair.left);
            }
            if let Some(previous) = nodes[pair.left].previous {
                self.candidate(text, nodes, previous, ranked, queue, rank_key)?;
            }
            self.candidate(text, nodes, pair.left, ranked, queue, rank_key)?;
        }
        let mut current = Some(0);
        while let Some(index) = current {
            let piece = &text[nodes[index].start..nodes[index].end];
            if let Some(&id) = self.lookup.get(piece) {
                output.push(id);
            } else if ranked && self.pre != "gemma4" {
                // The pinned BPE fallback looks up each raw UTF-8 byte of
                // the encoded symbol and emits only existing spellings. Some
                // vocabularies (Falcon's control bytes) omit alphabet symbols;
                // they do not reject those inputs. GGUF spellings are valid
                // UTF-8, so an isolated non-ASCII byte cannot have an entry.
                for byte in piece.bytes() {
                    let key = [byte];
                    if let Ok(key) = std::str::from_utf8(&key)
                        && let Some(&id) = self.lookup.get(key)
                    {
                        output.push(id);
                    }
                }
            } else {
                for &value in piece.as_bytes() {
                    output.push(self.byte(value)?);
                }
            }
            current = nodes[index].next;
        }
        Ok(())
    }

    fn candidate(
        &self,
        text: &str,
        nodes: &[Symbol],
        left: usize,
        ranked: bool,
        queue: &mut BinaryHeap<Pair>,
        rank_key: &mut String,
    ) -> Result<()> {
        let Some(right) = nodes[left].next else {
            return Ok(());
        };
        let priority = if ranked {
            rank_key.clear();
            rank_key.push_str(&text[nodes[left].start..nodes[left].end]);
            rank_key.push(' ');
            rank_key.push_str(&text[nodes[right].start..nodes[right].end]);
            let Some(&rank) = self.ranks.get(rank_key.as_str()) else {
                return Ok(());
            };
            -f64::from(u32::try_from(rank)?)
        } else {
            let Some(&id) = self.lookup.get(&text[nodes[left].start..nodes[right].end]) else {
                return Ok(());
            };
            f64::from(self.scores[usize::try_from(id)?])
        };
        queue.push(Pair {
            priority,
            left,
            right,
            size: nodes[right].end - nodes[left].start,
        });
        Ok(())
    }

    fn wordpiece(&self, text: &str, output: &mut Vec<u32>) -> Result<()> {
        let normalized = if self.strip_accents {
            text.nfd()
                .map(|value| value.0)
                .filter(|value| !unicode_normalization_alignments::char::is_combining_mark(*value))
                .collect::<String>()
        } else {
            text.to_owned()
        };
        let mut cleaned = String::new();
        for value in normalized.chars() {
            if value.is_whitespace() {
                cleaned.push(' ');
            } else if value != '\0' && value != '\u{fffd}' && !value.is_control() {
                let scalar = u32::from(value);
                let chinese = matches!(scalar, 0x4e00..=0x9fff | 0x3400..=0x4dbf | 0x20000..=0x2a6df | 0x2a700..=0x2b73f | 0x2b740..=0x2b81f | 0x2b920..=0x2ceaf | 0xf900..=0xfaff | 0x2f800..=0x2fa1f);
                if chinese {
                    cleaned.push(' ');
                }
                if self.lowercase {
                    cleaned.extend(value.to_lowercase());
                } else {
                    cleaned.push(value);
                }
                if chinese {
                    cleaned.push(' ');
                }
            }
        }
        let regex = &self.regexes[0];
        for word in regex.find_iter(&cleaned) {
            let word = format!("▁{}", word?.as_str());
            let start = output.len();
            let mut offset = 0;
            while offset < word.len() {
                let found = self
                    .prefixes
                    .matching(&self.raw, &word.as_bytes()[offset..])
                    .max_by_key(|&id| (self.raw[id].len(), id));
                if let Some(id) = found {
                    output.push(u32::try_from(id)?);
                    offset += self.raw[id].len();
                } else {
                    output.truncate(start);
                    output.push(
                        self.unknown
                            .ok_or_else(|| invalid("WordPiece unknown token unavailable"))?,
                    );
                    break;
                }
            }
        }
        Ok(())
    }

    fn unigram(&self, input: &str, output: &mut Vec<u32>) -> Result<()> {
        let mut text = self.charsmap.as_ref().map_or_else(
            || Ok(input.to_owned()),
            |charsmap| charsmap.normalize(input),
        )?;
        if self.remove_spaces {
            text = text
                .split(' ')
                .filter(|part| !part.is_empty())
                .collect::<Vec<_>>()
                .join(" ");
        }
        if self.prefix_space && !text.is_empty() {
            text.insert(0, ' ');
        }
        if self.escape_spaces {
            text = text.replace(' ', "▁");
        }
        if text.is_empty() {
            return Ok(());
        }
        let unknown = self
            .unknown
            .ok_or_else(|| invalid("Unigram unknown token unavailable"))?;
        let mut best = vec![(f64::NEG_INFINITY, 0_usize, unknown); text.len() + 1];
        best[0].0 = 0.0;
        for (offset, value) in text.char_indices() {
            let mut single = false;
            for index in self
                .prefixes
                .matching(&self.raw, &text.as_bytes()[offset..])
            {
                let token = &self.tokens[index];
                if !matches!(self.types[index], 1 | 4 | 5) {
                    continue;
                }
                let end = offset + token.len();
                single |= token.len() == value.len_utf8();
                let score = best[offset].0
                    + if self.types[index] == 4 {
                        0.0
                    } else {
                        f64::from(self.scores[index])
                    };
                if score > best[end].0 {
                    best[end] = (score, offset, u32::try_from(index)?);
                }
            }
            if !single {
                let end = offset + value.len_utf8();
                let score = best[offset].0 + f64::from(self.minimum_score - 10.0);
                if score > best[end].0 {
                    best[end] = (score, offset, unknown);
                }
            }
        }
        let mut reverse = Vec::new();
        let mut end = text.len();
        while end > 0 {
            let (_, start, id) = best[end];
            if start >= end {
                return Err(invalid("Unigram path unavailable"));
            }
            if id != unknown || reverse.last() != Some(&unknown) {
                reverse.push(id);
            }
            end = start;
        }
        output.extend(reverse.into_iter().rev());
        Ok(())
    }

    fn plamo(&self, input: &str, output: &mut Vec<u32>) -> Result<()> {
        let text = input.strip_prefix('\u{feff}').unwrap_or(input);
        let chars: Vec<_> = text.char_indices().collect();
        let mut best = vec![(i64::MAX / 2, 0_usize, None); chars.len() + 1];
        best[chars.len()].0 = 0;
        for index in (0..chars.len()).rev() {
            let (offset, value) = chars[index];
            best[index] = (best[index + 1].0 + 10_000_000, index + 1, None);
            for id in self
                .prefixes
                .matching(&self.raw, &text.as_bytes()[offset..])
            {
                let token = &self.tokens[id];
                let end = index + token.chars().count();
                if end > chars.len() {
                    continue;
                }
                let score =
                    self.plamo_scores[id].map_or_else(|| rounded_score(self.scores[id]), Ok)?;
                if score <= -20_000_000 {
                    continue;
                }
                let total = best[end].0 - i64::from(score);
                // The former length-sorted scan visited longer matches first.
                // Keep that tie preference without visiting unrelated tokens.
                if total < best[index].0
                    || (total == best[index].0 && best[index].2.is_some() && end > best[index].1)
                {
                    best[index] = (total, end, Some(u32::try_from(id)?));
                }
            }
            let _ = value;
        }
        let mut index = 0;
        while index < chars.len() {
            let (_, next, id) = best[index];
            if let Some(id) = id {
                output.push(id);
            } else {
                let mut storage = [0; 4];
                for &value in chars[index].1.encode_utf8(&mut storage).as_bytes() {
                    output.push(self.byte(value)?);
                }
            }
            index = next;
        }
        Ok(())
    }

    fn dna(&self, input: &str, output: &mut Vec<u32>) -> Result<()> {
        let mut position = 0;
        while let Some(start) = input[position..].find("<dna>") {
            let start = position + start;
            if start > position {
                self.bpe(&input[position..start], output)?;
            }
            output.push(
                *self
                    .lookup
                    .get("<dna>")
                    .ok_or_else(|| invalid("DNA begin token unavailable"))?,
            );
            let begin = start + 5;
            let end = input[begin..]
                .find("</dna>")
                .map_or(input.len(), |v| begin + v);
            let sequence = input[begin..end].to_ascii_uppercase();
            for chunk in sequence.as_bytes().chunks(6) {
                let mut kmer = std::str::from_utf8(chunk)?.to_owned();
                while kmer.len() < 6 {
                    kmer.push('A');
                }
                kmer.push('\u{e000}');
                output.push(
                    self.lookup
                        .get(&kmer)
                        .or_else(|| self.lookup.get("<oov>"))
                        .copied()
                        .ok_or_else(|| invalid("DNA fallback unavailable"))?,
                );
            }
            if end == input.len() {
                return Ok(());
            }
            output.push(
                *self
                    .lookup
                    .get("</dna>")
                    .ok_or_else(|| invalid("DNA end token unavailable"))?,
            );
            position = end + 6;
        }
        if position < input.len() {
            self.bpe(&input[position..], output)?;
        }
        Ok(())
    }

    /// Returns the bytes of one token, with explicit control rendering.
    ///
    /// # Errors
    /// Rejects an invalid token ID or malformed byte notation/alphabet.
    pub fn piece(&self, id: u32, special: bool) -> Result<Vec<u8>> {
        let mut output = Vec::new();
        self.append_piece(id, special, &mut output)?;
        Ok(output)
    }

    // Whole-sequence decode shares its destination across pieces. In particular,
    // exhaustive roundtrip checks must not allocate a temporary Vec per token.
    fn append_piece(&self, id: u32, special: bool, output: &mut Vec<u8>) -> Result<()> {
        let index = usize::try_from(id)?;
        let token = self
            .tokens
            .get(index)
            .ok_or_else(|| invalid("tokenizer piece ID out of range"))?;
        let kind = self.types[index];
        if matches!(kind, 2 | 3) {
            if special {
                output.extend_from_slice(token.as_bytes());
            }
            return Ok(());
        }
        if kind == 5 {
            return Ok(());
        }
        if kind == 4 {
            output.extend_from_slice(token.as_bytes());
            return Ok(());
        }
        if kind == 6 {
            output.push(parse_byte(token)?);
            return Ok(());
        }
        match self.family {
            Family::Bpe if self.pre != "gemma4" && self.model != "whitespace" && kind == 1 => {
                for character in token.chars() {
                    output.push(decode_character(character)?);
                }
            }
            Family::Rwkv => output.extend_from_slice(&self.raw[index]),
            Family::Spm | Family::Unigram | Family::WordPiece => {
                append_spaces(token, output);
            }
            _ => append_spaces(token.trim_end_matches('\u{e000}'), output),
        }
        Ok(())
    }

    /// Concatenates decoded pieces using model boundary and cleanup policy.
    ///
    /// # Errors
    /// Rejects invalid IDs or more than 1 MiB of decoded output.
    pub fn decode(
        &self,
        ids: &[u32],
        remove_special: bool,
        unparse_special: bool,
    ) -> Result<Vec<u8>> {
        let mut ids = ids;
        if remove_special && self.add_bos && ids.first().copied() == self.bos {
            ids = &ids[1..];
        }
        if remove_special && self.add_eos && ids.last().copied() == self.eos {
            ids = &ids[..ids.len() - 1];
        }
        let mut output = Vec::new();
        for &id in ids {
            self.append_piece(id, unparse_special, &mut output)?;
            if output.len() > LIMIT {
                return Err(invalid("tokenizer decoded bytes exceed bound"));
            }
        }
        if self.prefix_space && output.first() == Some(&b' ') {
            output.remove(0);
        }
        Ok(output)
    }
}

fn append_spaces(token: &str, output: &mut Vec<u8>) {
    for (index, fragment) in token.split('▁').enumerate() {
        if index != 0 {
            output.push(b' ');
        }
        output.extend_from_slice(fragment.as_bytes());
    }
}

fn rounded_score(score: f32) -> Result<i32> {
    let value = f64::from(score) * 10_000.0;
    value.round().to_string().parse().map_err(Into::into)
}
fn parse_byte(token: &str) -> Result<u8> {
    u8::from_str_radix(
        token
            .strip_prefix("<0x")
            .and_then(|v| v.strip_suffix('>'))
            .filter(|v| v.len() == 2)
            .ok_or_else(|| invalid("byte token notation malformed"))?,
        16,
    )
    .map_err(Into::into)
}
fn byte_alphabet() -> Result<[char; 256]> {
    let mut table = ['\0'; 256];
    let mut next = 256;
    for value in 0_u16..256 {
        let byte = u8::try_from(value)?;
        table[usize::from(byte)] = if matches!(byte, 33..=126 | 161..=172 | 174..=255) {
            char::from(byte)
        } else {
            let value =
                char::from_u32(next).ok_or_else(|| invalid("byte alphabet scalar invalid"))?;
            next += 1;
            value
        };
    }
    Ok(table)
}

fn unescape_rwkv(token: &str) -> Result<Vec<u8>> {
    let mut output = Vec::new();
    let mut bytes = token.bytes();
    while let Some(value) = bytes.next() {
        if value != b'\\' {
            output.push(value);
            continue;
        }
        match bytes.next() {
            Some(b'n') => output.push(b'\n'),
            Some(b'r') => output.push(b'\r'),
            Some(b't') => output.push(b'\t'),
            Some(b'x') => {
                let hex = [
                    bytes.next().ok_or_else(|| invalid("RWKV truncated hex"))?,
                    bytes.next().ok_or_else(|| invalid("RWKV truncated hex"))?,
                ];
                output.push(u8::from_str_radix(std::str::from_utf8(&hex)?, 16)?);
            }
            Some(value) => output.push(value),
            None => return Err(invalid("RWKV trailing escape")),
        }
    }
    Ok(output)
}

#[allow(
    clippy::unicode_not_nfc,
    reason = "Preserve the pinned upstream regex scalar ranges exactly"
)]
fn patterns(pre: &str) -> Result<Vec<&'static str>> {
    let gpt = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)";
    let llama = r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
    Ok(match pre {
        "qwen35" => vec![],
        "gpt-2" | "phi-2" | "mpt" | "olmo" | "jais" | "jina-v2-es" | "jina-v2-de"
        | "jina-v2-code" | "jina-v1-en" | "roberta-bpe" => vec![gpt],
        "llama3" | "llama-v3" | "llama-bpe" | "smaug-bpe" => vec![llama],
        "qwen2" | "stablelm2" => vec![
            r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
        ],
        "starcoder" | "refact" | "command-r" => vec![r"\p{N}", gpt],
        "falcon" => vec![r"[\p{P}\$\+<=>\^~\|`]+", gpt, r"[0-9][0-9][0-9]"],
        "deepseek-coder" => vec![
            r"[\r\n]",
            r"\s?\p{L}+",
            r"\s?\p{P}+",
            "[一-龥ࠀ-一가-퟿]+",
            r"\p{N}",
        ],
        "deepseek-llm" => vec![
            r"[\r\n]",
            "\\s?[A-Za-zµÀ-ÖØ-öø-ƺƼ-ƿǄ-ʓʕ-ʯͰ-ͳͶͷͻ-ͽͿΆΈ-ΊΌΎ-ΡΣ-ϵϷ-ҁҊ-ԯԱ-ՖႠ-ჅᎠ-Ᏽᏸ-ᏽᲐ-ᲺᲽ-Ჿᴀ-ᴫᵫ-ᵷᵹ-ᶚḀ-ἕἘ-Ἕἠ-ὅὈ-Ὅὐ-ὗὙὛὝὟ-ώᾀ-ᾴᾶ-ᾼιῂ-ῄῆ-ῌῐ-ΐῖ-Ίῠ-Ῥῲ-ῴῶ-ῼℂℇℊ-ℓℕℙ-ℝℤΩℨK-ℭℯ-ℴℹℼ-ℿⅅ-ⅉⅎↃↄⰀ-ⱻⱾ-ⳤⳫ-ⳮⳲⳳꙀ-ꙭꚀ-ꚛꜢ-ꝯꝱ-ꞇꞋ-ꞎꭰ-ꮿﬀ-ﬆﬓ-ﬗＡ-Ｚａ-ｚ𐐀-𐑏𐒰-𐓓𐓘-𐓻𐲀-𐲲𐳀-𐳲𑢠-𑣟𞤀-𞥃]+",
            "\\s?[!-/:-~！-／：-～‘-‟　-。]+",
            r"\s+$",
            "[一-龥ࠀ-一가-퟿]+",
            r"\p{N}+",
        ],
        "poro-chat" | "bloom" => vec![r" ?[^(\s|.,!?…。，、।۔،)]+"],
        "viking" => vec![r" ?[^(\s|.,!?…。，、।۔،)]+", r"\p{N}"],
        "gemma4" => vec![r"[^\n]+|[\n]+"],
        "whitespace" => vec![r"\S+"],
        "default" => vec![r"[\p{P}\$\+<=>\^~\|]+", gpt, r"\p{N}+", r"[0-9][0-9][0-9]"],
        other => return Err(invalid(format!("unsupported BPE pretokenizer {other}"))),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokenizer(model: &str, tokens: &[&str], scores: &[f32]) -> Result<GgufTokenizer> {
        let metadata = Metadata::from([
            ("tokenizer.ggml.model".into(), model.into()),
            ("tokenizer.ggml.tokens".into(), tokens.to_vec().into()),
            ("tokenizer.ggml.merges".into(), Vec::<String>::new().into()),
            ("tokenizer.ggml.scores".into(), scores.to_vec().into()),
            ("tokenizer.ggml.unknown_token_id".into(), 0_u32.into()),
            ("tokenizer.ggml.add_space_prefix".into(), false.into()),
            ("tokenizer.ggml.add_bos_token".into(), false.into()),
            ("tokenizer.ggml.add_eos_token".into(), false.into()),
            ("tokenizer.ggml.normalizer.lowercase".into(), false.into()),
            (
                "tokenizer.ggml.normalizer.strip_accents".into(),
                false.into(),
            ),
        ]);
        GgufTokenizer::from_metadata(&metadata, tokens.len())
    }

    #[test]
    fn prefix_ranges_retain_nested_and_duplicate_matches() {
        let raw = ["", "a", "abc", "ab", "abc", "abx", "b", "é", "éa", "ê"]
            .map(|token| token.as_bytes().to_vec());
        let prefixes = PrefixIndex::new(&raw);
        for (input, expected) in [
            ("abcd", vec![1, 3, 2, 4]),
            ("abx", vec![1, 3, 5]),
            ("éab", vec![7, 8]),
            ("ê", vec![9]),
            ("z", vec![]),
            ("", vec![]),
        ] {
            assert_eq!(
                prefixes
                    .matching(&raw, input.as_bytes())
                    .collect::<Vec<_>>(),
                expected
            );
        }
    }

    #[test]
    fn special_search_keeps_earliest_longest_id_and_literal_policy() {
        let tokens = ["", "a", "aba", "aba", "b", "é", "éa", "aé", "\0"].map(str::to_owned);
        let types = [4, 4, 3, 4, 2, 3, 4, 4, 4];
        let fragments: &[&[u8]] = &[b"", b"a", b"ab", b"b", "é".as_bytes(), b"\0", b"\xff"];
        for parse_special in [false, true] {
            let mut index = SpecialIndex::new(&tokens, &types, parse_special);
            assert!(index.matcher.is_some());
            for use_matcher in [true, false] {
                if !use_matcher {
                    index.matcher = None;
                }
                for left in fragments {
                    for middle in fragments {
                        for right in fragments {
                            let input = [*left, *middle, *right].concat();
                            let expected = tokens
                                .iter()
                                .enumerate()
                                .filter(|(id, token)| {
                                    !token.is_empty() && (types[*id] == 4 || parse_special)
                                })
                                .filter_map(|(id, token)| {
                                    input
                                        .windows(token.len())
                                        .position(|part| part == token.as_bytes())
                                        .map(|offset| (offset, std::cmp::Reverse(token.len()), id))
                                })
                                .min()
                                .map(|(offset, _, id)| (offset, id));
                            assert_eq!(index.find(&tokens, &input), expected, "{input:?}");
                        }
                    }
                }
            }
        }
        let empty = SpecialIndex::new(&tokens, &[1; 9], true);
        assert!(empty.matcher.is_none());
        assert_eq!(empty.find(&tokens, b"aba"), None);
        let oversized = ["x".repeat(LIMIT + 1)];
        let linear = SpecialIndex::new(&oversized, &[4], true);
        assert!(linear.matcher.is_none());
        assert_eq!(
            linear.find(&oversized, oversized[0].as_bytes()),
            Some((0, 0))
        );
    }

    #[test]
    fn decoding_appends_family_pieces_and_retains_lazy_errors() -> Result<()> {
        for model in ["llama", "t5", "bert", "plamo2"] {
            let tokens = ["?", "▁a", "▁▁é▁", "<C>", "<U>", "<unused>", "<0xFF>"];
            let mut tokenizer = tokenizer(model, &tokens, &[0.0; 7])?;
            tokenizer.types = vec![1, 1, 1, 3, 4, 5, 6];
            for special in [false, true] {
                let expected = if special {
                    b"? a  \xc3\xa9 <C><U>\xff".as_slice()
                } else {
                    b"? a  \xc3\xa9 <U>\xff".as_slice()
                };
                assert_eq!(
                    tokenizer.decode(&[0, 1, 2, 3, 4, 5, 6], false, special)?,
                    expected
                );
                assert_eq!(tokenizer.piece(2, special)?, "  é ".as_bytes());
                assert!(tokenizer.decode(&[1, 7], false, special).is_err());
            }
            tokenizer.tokens[6] = "<0xGG>".into();
            assert!(tokenizer.decode(&[1, 6], false, false).is_err());
            assert_eq!(tokenizer.decode(&[1], false, false)?, b" a");
        }
        let bpe = tokenizer("gpt2", &["?", "Ġa", "é"], &[0.0; 3])?;
        assert_eq!(bpe.decode(&[1, 2], false, false)?, b" a\xe9");
        let rwkv = tokenizer("rwkv", &["?", "\\xFF", "é"], &[0.0; 3])?;
        assert_eq!(rwkv.decode(&[1, 2], false, false)?, b"\xff\xc3\xa9");
        Ok(())
    }

    #[test]
    fn bpe_missing_symbols_follow_raw_byte_lookup_without_inventing_tokens() -> Result<()> {
        let bpe = tokenizer("gpt2", &["?", "ā", "a", "b", "Ġ"], &[0.0; 5])?;
        assert_eq!(bpe.encode(b"\x01\x03", false, false)?, [1]);
        assert_eq!(bpe.encode(b"\x03", false, false)?, Vec::<u32>::new());
        // A merge can name a symbol absent from the vocabulary. The source
        // fallback then queries raw bytes, rather than splitting back into
        // existing Unicode alphabet symbols (which would insert token4).
        let mut bpe = bpe;
        bpe.ranks.insert("a Ġ".into(), 0);
        let mut output = Vec::new();
        bpe.merge("aĠb", true, &mut output)?;
        assert_eq!(output, [2, 3]);
        Ok(())
    }

    #[test]
    fn indexed_tokenizers_keep_family_specific_duplicate_and_path_ties() -> Result<()> {
        let rwkv = tokenizer(
            "rwkv",
            &["", "a", "abc", "ab", "abc", "\\xFF", "é"],
            &[0.0; 7],
        )?;
        assert_eq!(
            rwkv.encode("abcabaé".as_bytes(), false, false)?,
            [2, 3, 1, 6]
        );
        let mut raw = Vec::new();
        rwkv.greedy(b"\xffabc", &mut raw)?;
        assert_eq!(raw, [5, 2]);

        let wordpiece = tokenizer(
            "bert",
            &["<unk>", "▁", "▁a", "▁ab", "▁ab", "b", "!", "▁!", "▁é"],
            &[0.0; 9],
        )?;
        assert_eq!(wordpiece.encode(b"ab!", false, false)?, [4, 7]);
        assert_eq!(wordpiece.encode(b"abc!", false, false)?, [0, 7]);
        assert_eq!(wordpiece.encode("é".as_bytes(), false, false)?, [8]);

        let tokens = ["?", "a", "b", "ab", "ab", "é"];
        let unigram = tokenizer("t5", &tokens, &[-2.0, -1.0, -1.0, -2.0, -2.0, -1.0])?;
        assert_eq!(unigram.encode("abéx".as_bytes(), false, false)?, [3, 5, 0]);
        let scored = tokenizer("t5", &tokens, &[-2.0, -1.0, -1.0, -2.0, -1.0, -1.0])?;
        assert_eq!(scored.encode(b"ab", false, false)?, [4]);

        let plamo = tokenizer(
            "plamo2",
            &["?", "a", "b", "ab", "ab", "é", "<0xC3>", "<0xA9>"],
            &[0.0, 0.0, 0.0, 0.0, 0.0, -1000.0, 0.0, 0.0],
        )?;
        assert_eq!(plamo.encode("abaé".as_bytes(), false, false)?, [3, 1, 6, 7]);
        Ok(())
    }

    #[test]
    fn prepared_byte_tables_keep_alphabet_and_fallback_priority() -> Result<()> {
        let alphabet = byte_alphabet()?;
        for value in 0..=u8::MAX {
            assert_eq!(decode_character(alphabet[usize::from(value)])?, value);
        }
        let tokenizer = tokenizer("llama", &["?", "A", "<0x41>"], &[0.0; 3])?;
        assert_eq!(tokenizer.byte(b'A')?, 2);
        assert!(tokenizer.byte(b'B').is_err());
        Ok(())
    }

    #[test]
    fn plamo_score_preparation_preserves_lazy_out_of_range_errors() -> Result<()> {
        for score in [f32::MIN, f32::MAX] {
            let tokenizer = tokenizer("plamo2", &["?", "a", "b"], &[0.0, score, 0.0])?;
            assert_eq!(tokenizer.encode(b"b", false, false)?, [2]);
            assert!(tokenizer.encode(b"a", false, false).is_err());
        }
        Ok(())
    }
}
