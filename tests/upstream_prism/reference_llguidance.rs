//! Clean-room independent safe Rust `LLGuidance` grammar/schema engine and candidate buffer provider.
//!
//! No `bonsai_prism_runtime`, `llguidance`, or native bridge code is imported or called.
//! Compiles Lark grammars and order-preserving `%json` schemas to byte-level CFG rules and
//! runs incremental Earley traversal over vocabulary byte tries while preserving raw candidate
//! bit patterns (`-0.0`, quiet/signaling NaN payloads).

#[path = "reference_llguidance/candidates.rs"]
mod candidates;
#[path = "reference_llguidance/grammar.rs"]
mod grammar;
#[path = "reference_llguidance/json_ast.rs"]
mod json_ast;

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use serde_json::{Value, json};

use super::llama_grammar::{exact_fields, field, uint};
use super::reference_vocab::ReferenceVocab;
use super::{Result, invalid};
use candidates::{CandidateBuffer, CopyEncoding};
use grammar::{EarleyState, Grammar};

pub(super) const ID: &str = "tests/test-grammar-llguidance.cpp";
const SOURCE_SHA256: &str = "050d42e1577bdb537bd8c133e3a71354d5b64a9ea5cc8606962531095e28d56f";

#[derive(Default)]
struct TrieNode {
    edges: Vec<(u8, usize)>,
    tokens: Vec<usize>,
}

/// Pre-indexed vocabulary byte trie and end-of-generation mask for fast candidate filtering.
pub(super) struct VocabTrie {
    vocab: Arc<ReferenceVocab>,
    nodes: Vec<TrieNode>,
    eog_ids: Vec<usize>,
}

impl VocabTrie {
    pub(super) fn build(vocab: Arc<ReferenceVocab>) -> Result<Self> {
        let count = vocab.vocab_size();
        let mut nodes = vec![TrieNode::default()];
        let mut eog_ids = Vec::new();
        for raw_idx in 0..count {
            let tok = i32::try_from(raw_idx)?;
            if vocab.is_eog(tok) {
                eog_ids.push(raw_idx);
                continue;
            }
            let piece = vocab.token_to_piece(tok, 0, false)?;
            if piece.is_empty() {
                continue;
            }
            let mut cur = 0_usize;
            for &byte in &piece {
                let next = if let Some(&(_, child)) =
                    nodes[cur].edges.iter().find(|&&(edge_b, _)| edge_b == byte)
                {
                    child
                } else {
                    let id = nodes.len();
                    nodes.push(TrieNode::default());
                    nodes[cur].edges.push((byte, id));
                    id
                };
                cur = next;
            }
            nodes[cur].tokens.push(raw_idx);
        }
        Ok(Self {
            vocab,
            nodes,
            eog_ids,
        })
    }

    fn allowed_tokens(&self, base: &EarleyState<'_>) -> Result<Vec<bool>> {
        let mut allowed = vec![false; self.vocab.vocab_size()];
        if base.accepts_full() {
            for &idx in &self.eog_ids {
                allowed[idx] = true;
            }
        }
        // Snapshot the current Earley state into an owned stack for bounded trie traversal.
        let mut walker = base.clone();
        let mut stack: Vec<(usize, usize)> = vec![(0, 0)];
        while let Some((node_idx, edge_cursor)) = stack.last_mut() {
            let n = *node_idx;
            let e = *edge_cursor;
            if e >= self.nodes[n].edges.len() {
                stack.pop();
                if !stack.is_empty() {
                    walker.pop_byte();
                }
                continue;
            }
            *edge_cursor += 1;
            let (byte, child_idx) = self.nodes[n].edges[e];
            if walker.push_byte(byte)? {
                for &tok_idx in &self.nodes[child_idx].tokens {
                    allowed[tok_idx] = true;
                }
                if self.nodes[child_idx].edges.is_empty() {
                    walker.pop_byte();
                } else {
                    stack.push((child_idx, 0));
                }
            }
        }
        Ok(allowed)
    }
}

struct GrammarSampler {
    grammar: Grammar,
    accepted: Vec<u8>,
    failed: bool,
    trie: Arc<VocabTrie>,
}

impl GrammarSampler {
    fn reset(&mut self) {
        self.accepted.clear();
        self.failed = false;
    }

    fn accept(&mut self, token: i32) -> Result<()> {
        if self.failed {
            return Ok(());
        }
        if self.trie.vocab.is_eog(token) {
            return Ok(());
        }
        let piece = self.trie.vocab.token_to_piece(token, 0, false)?;
        let mut state = self.current_state()?;
        for &b in &piece {
            if !state.push_byte(b)? {
                self.failed = true;
                break;
            }
        }
        self.accepted.extend_from_slice(&piece);
        Ok(())
    }

    fn current_state(&self) -> Result<EarleyState<'_>> {
        let mut state = EarleyState::new(&self.grammar)?;
        for &b in &self.accepted {
            if !state.push_byte(b)? {
                return Err(invalid("accepted prefix failed internally"));
            }
        }
        Ok(state)
    }

    fn apply(&self, buf: &mut CandidateBuffer) -> Result<()> {
        let end = buf
            .offset
            .checked_add(buf.size)
            .filter(|&e| e <= buf.entries.len())
            .ok_or_else(|| invalid("LLG candidate slice out of bounds"))?;
        if self.failed {
            for entry in &mut buf.entries[buf.offset..end] {
                entry.logit_bits = f32::NEG_INFINITY.to_bits();
            }
            return Ok(());
        }
        let state = self.current_state()?;
        let allowed = self.trie.allowed_tokens(&state)?;
        for entry in &mut buf.entries[buf.offset..end] {
            let permit = usize::try_from(entry.id)
                .ok()
                .and_then(|idx| allowed.get(idx).copied())
                .unwrap_or(false);
            if !permit {
                entry.logit_bits = f32::NEG_INFINITY.to_bits();
            }
        }
        Ok(())
    }
}

struct DistSampler {
    rng_state: u64,
}

impl DistSampler {
    const fn new(seed: u64) -> Self {
        Self {
            rng_state: seed ^ 0x9e37_79b9_7f4a_7c15,
        }
    }

    fn next_f64(&mut self) -> f64 {
        let mut x = self.rng_state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        self.rng_state = x;
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        x ^= x >> 31;
        let mantissa = x >> 11;
        // Divide 53-bit integer by 2^53 cleanly via f64::from_bits exponent normalization or exact scale.
        let units = u32::try_from(mantissa >> 22).unwrap_or(0);
        f64::from(units) / f64::from(1_u32 << 31)
    }

    fn apply(&mut self, buf: &mut CandidateBuffer) -> Result<()> {
        let end = buf
            .offset
            .checked_add(buf.size)
            .filter(|&e| e <= buf.entries.len())
            .ok_or_else(|| invalid("LLG dist candidate slice out of bounds"))?;
        let slice = &mut buf.entries[buf.offset..end];
        if slice.is_empty() {
            return Err(invalid("LLG dist cannot sample from empty candidate slice"));
        }
        let mut max_logit = f32::NEG_INFINITY;
        for entry in slice.iter() {
            let v = f32::from_bits(entry.logit_bits);
            if v > max_logit {
                max_logit = v;
            }
        }
        let mut weights = Vec::with_capacity(slice.len());
        let mut total = 0.0_f32;
        for entry in slice.iter() {
            let v = f32::from_bits(entry.logit_bits);
            let w = if v.is_nan() || v == f32::NEG_INFINITY || !max_logit.is_finite() {
                0.0_f32
            } else {
                (v - max_logit).exp()
            };
            weights.push(w);
            total += w;
        }
        let mut chosen = 0_usize;
        let mut picked = false;
        if total > 0.0 {
            let draw = self.next_f64() * f64::from(total);
            let mut accum = 0.0_f64;
            for (idx, (&w, entry)) in weights.iter().zip(slice.iter_mut()).enumerate() {
                let p_f32 = w / total;
                entry.prob_bits = p_f32.to_bits();
                accum += f64::from(w);
                if w > 0.0 && !picked {
                    chosen = idx;
                    if accum >= draw {
                        picked = true;
                    }
                }
            }
        }
        buf.selected = i64::try_from(chosen)?;
        Ok(())
    }
}

enum Sampler {
    Grammar(Box<GrammarSampler>),
    Dist(DistSampler),
    Chain(Vec<u64>),
}

/// Self-contained `LLGuidance` session state stored per `ReferenceModelSession`.
pub(super) struct LlgRegistry {
    next_id: u64,
    candidates: BTreeMap<u64, CandidateBuffer>,
    samplers: BTreeMap<u64, Sampler>,
    trie_cache: BTreeMap<usize, Arc<VocabTrie>>,
}

impl LlgRegistry {
    pub(super) const fn new() -> Self {
        Self {
            next_id: 1,
            candidates: BTreeMap::new(),
            samplers: BTreeMap::new(),
            trie_cache: BTreeMap::new(),
        }
    }

    const fn alloc(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    fn trie_for(&mut self, vocab: &Arc<ReferenceVocab>) -> Result<Arc<VocabTrie>> {
        let key = Arc::as_ptr(vocab).addr();
        if let Some(cached) = self.trie_cache.get(&key) {
            return Ok(Arc::clone(cached));
        }
        let built = Arc::new(VocabTrie::build(Arc::clone(vocab))?);
        self.trie_cache.insert(key, Arc::clone(&built));
        Ok(built)
    }

    pub(super) fn vocabulary_info(vocab: &ReferenceVocab) -> Value {
        json!({
            "Count": vocab.vocab_size(),
            "Eot": vocab.eot.unwrap_or(-1),
            "Eos": vocab.eos.unwrap_or(-1),
        })
    }

    pub(super) fn candidates_new(&mut self, operation: &Value) -> Result<Value> {
        exact_fields(operation, &["Op", "Count", "File", "Selected", "Sorted"])?;
        let count = usize::try_from(uint(operation, "Count")?)?;
        let path_str = field(operation, "File")?
            .as_str()
            .ok_or_else(|| invalid("LLG File must be path string"))?;
        let selected = field(operation, "Selected")?
            .as_i64()
            .ok_or_else(|| invalid("LLG Selected must be integer"))?;
        let sorted = field(operation, "Sorted")?
            .as_bool()
            .ok_or_else(|| invalid("LLG Sorted must be boolean"))?;
        let buf = CandidateBuffer::from_file(Path::new(path_str), count, selected, sorted)?;
        let id = self.alloc();
        self.candidates.insert(id, buf);
        Ok(json!({ "Candidates": id }))
    }

    pub(super) fn candidates_clone(&mut self, operation: &Value) -> Result<Value> {
        exact_fields(operation, &["Op", "Source"])?;
        let src_id = uint(operation, "Source")?;
        let src = self
            .candidates
            .get(&src_id)
            .ok_or_else(|| invalid("unknown LLG source candidates handle"))?
            .clone();
        let id = self.alloc();
        self.candidates.insert(id, src);
        Ok(json!({ "Candidates": id }))
    }

    pub(super) fn candidates_load(&mut self, operation: &Value) -> Result<Value> {
        exact_fields(
            operation,
            &["Op", "Candidates", "File", "Encoding", "Metadata"],
        )?;
        let cid = uint(operation, "Candidates")?;
        let path_str = field(operation, "File")?
            .as_str()
            .ok_or_else(|| invalid("LLG File must be path string"))?;
        let enc_str = field(operation, "Encoding")?
            .as_str()
            .ok_or_else(|| invalid("LLG Encoding must be string"))?;
        let encoding = CopyEncoding::parse(enc_str)?;
        let metadata = field(operation, "Metadata")?;
        let buf = self
            .candidates
            .get_mut(&cid)
            .ok_or_else(|| invalid("unknown LLG candidates handle"))?;
        buf.load_file(Path::new(path_str), encoding, metadata)?;
        Ok(Value::Null)
    }

    pub(super) fn candidates_copy(&mut self, operation: &Value) -> Result<Value> {
        exact_fields(
            operation,
            &["Op", "Candidates", "Source", "Encoding", "Metadata"],
        )?;
        let dst_id = uint(operation, "Candidates")?;
        let src_id = uint(operation, "Source")?;
        let enc_str = field(operation, "Encoding")?
            .as_str()
            .ok_or_else(|| invalid("LLG Encoding must be string"))?;
        let encoding = CopyEncoding::parse(enc_str)?;
        let metadata = field(operation, "Metadata")?;
        let src = self
            .candidates
            .get(&src_id)
            .ok_or_else(|| invalid("unknown LLG source candidates handle"))?
            .clone();
        let dst = self
            .candidates
            .get_mut(&dst_id)
            .ok_or_else(|| invalid("unknown LLG destination candidates handle"))?;
        dst.copy_from(&src, encoding, metadata)?;
        Ok(Value::Null)
    }

    pub(super) fn candidates_patch(&mut self, operation: &Value) -> Result<Value> {
        exact_fields(operation, &["Op", "Candidates", "Patches"])?;
        let cid = uint(operation, "Candidates")?;
        let patches = field(operation, "Patches")?
            .as_array()
            .ok_or_else(|| invalid("LLG Patches must be array"))?;
        let buf = self
            .candidates
            .get_mut(&cid)
            .ok_or_else(|| invalid("unknown LLG candidates handle"))?;
        buf.patch(patches)?;
        Ok(Value::Null)
    }

    pub(super) fn candidates_inspect(&self, operation: &Value) -> Result<Value> {
        exact_fields(operation, &["Op", "Candidates", "Indices"])?;
        let cid = uint(operation, "Candidates")?;
        let indices = parse_indices(field(operation, "Indices")?)?;
        let buf = self
            .candidates
            .get(&cid)
            .ok_or_else(|| invalid("unknown LLG candidates handle"))?;
        buf.snapshot(&indices)
    }

    pub(super) fn candidates_free(&mut self, operation: &Value) -> Result<Value> {
        exact_fields(operation, &["Op", "Candidates"])?;
        let cid = uint(operation, "Candidates")?;
        self.candidates
            .remove(&cid)
            .ok_or_else(|| invalid("unknown LLG candidates handle"))?;
        Ok(Value::Null)
    }

    pub(super) fn sampler_new_grammar(
        &mut self,
        operation: &Value,
        vocab: &Arc<ReferenceVocab>,
    ) -> Result<Value> {
        exact_fields(operation, &["Op", "Model", "GrammarKind", "GrammarData"])?;
        let kind = field(operation, "GrammarKind")?
            .as_str()
            .ok_or_else(|| invalid("LLG GrammarKind must be string"))?;
        if kind != "lark" {
            return Err(invalid(format!("unsupported LLG GrammarKind {kind}")));
        }
        let data = field(operation, "GrammarData")?
            .as_str()
            .ok_or_else(|| invalid("LLG GrammarData must be string"))?;
        let compiled = grammar::compile_lark(data)?;
        let trie = self.trie_for(vocab)?;
        let id = self.alloc();
        self.samplers.insert(
            id,
            Sampler::Grammar(Box::new(GrammarSampler {
                grammar: compiled,
                accepted: Vec::new(),
                failed: false,
                trie,
            })),
        );
        Ok(json!({ "Sampler": id }))
    }

    pub(super) fn sampler_new_dist(&mut self, operation: &Value) -> Result<Value> {
        exact_fields(operation, &["Op", "Seed"])?;
        let seed = uint(operation, "Seed")?;
        let id = self.alloc();
        self.samplers
            .insert(id, Sampler::Dist(DistSampler::new(seed)));
        Ok(json!({ "Sampler": id }))
    }

    pub(super) fn sampler_new_chain(&mut self, operation: &Value) -> Result<Value> {
        exact_fields(operation, &["Op", "NoPerf"])?;
        let _ = field(operation, "NoPerf")?
            .as_bool()
            .ok_or_else(|| invalid("LLG NoPerf must be boolean"))?;
        let id = self.alloc();
        self.samplers.insert(id, Sampler::Chain(Vec::new()));
        Ok(json!({ "Sampler": id }))
    }

    pub(super) fn sampler_chain_add(&mut self, operation: &Value) -> Result<Value> {
        exact_fields(operation, &["Op", "Chain", "Sampler"])?;
        let chain_id = uint(operation, "Chain")?;
        let child_id = uint(operation, "Sampler")?;
        if !self.samplers.contains_key(&child_id) {
            return Err(invalid("unknown child LLG sampler handle"));
        }
        let Some(Sampler::Chain(children)) = self.samplers.get_mut(&chain_id) else {
            return Err(invalid("unknown or non-chain LLG handle"));
        };
        children.push(child_id);
        Ok(Value::Null)
    }

    pub(super) fn sampler_reset(&mut self, operation: &Value) -> Result<Value> {
        exact_fields(operation, &["Op", "Sampler"])?;
        let sid = uint(operation, "Sampler")?;
        let children = match self.samplers.get_mut(&sid) {
            Some(Sampler::Grammar(g)) => {
                g.reset();
                return Ok(Value::Null);
            }
            Some(Sampler::Dist(_)) => return Ok(Value::Null),
            Some(Sampler::Chain(children)) => children.clone(),
            None => return Err(invalid("unknown LLG sampler handle")),
        };
        for child_id in children {
            if let Some(Sampler::Grammar(g)) = self.samplers.get_mut(&child_id) {
                g.reset();
            }
        }
        Ok(Value::Null)
    }

    pub(super) fn sampler_accept(&mut self, operation: &Value) -> Result<Value> {
        exact_fields(operation, &["Op", "Sampler", "Token"])?;
        let sid = uint(operation, "Sampler")?;
        let tok = i32::try_from(
            field(operation, "Token")?
                .as_i64()
                .ok_or_else(|| invalid("LLG Token must be integer"))?,
        )?;
        let children = match self.samplers.get_mut(&sid) {
            Some(Sampler::Grammar(g)) => {
                g.accept(tok)?;
                return Ok(Value::Null);
            }
            Some(Sampler::Dist(_)) => return Ok(Value::Null),
            Some(Sampler::Chain(children)) => children.clone(),
            None => return Err(invalid("unknown LLG sampler handle")),
        };
        for child_id in children {
            if let Some(Sampler::Grammar(g)) = self.samplers.get_mut(&child_id) {
                g.accept(tok)?;
            }
        }
        Ok(Value::Null)
    }

    pub(super) fn sampler_apply(&mut self, operation: &Value) -> Result<Value> {
        exact_fields(operation, &["Op", "Sampler", "Candidates", "Indices"])?;
        let sid = uint(operation, "Sampler")?;
        let cid = uint(operation, "Candidates")?;
        let indices = parse_indices(field(operation, "Indices")?)?;
        let mut buf = self
            .candidates
            .get(&cid)
            .ok_or_else(|| invalid("unknown LLG candidates handle"))?
            .clone();
        match self.samplers.get_mut(&sid) {
            Some(Sampler::Grammar(g)) => g.apply(&mut buf)?,
            Some(Sampler::Dist(d)) => d.apply(&mut buf)?,
            Some(Sampler::Chain(children)) => {
                let ids = children.clone();
                for child_id in ids {
                    match self.samplers.get_mut(&child_id) {
                        Some(Sampler::Grammar(g)) => g.apply(&mut buf)?,
                        Some(Sampler::Dist(d)) => d.apply(&mut buf)?,
                        _ => return Err(invalid("invalid child sampler inside chain")),
                    }
                }
            }
            None => return Err(invalid("unknown LLG sampler handle")),
        }
        let snap = buf.snapshot(&indices)?;
        self.candidates.insert(cid, buf);
        Ok(snap)
    }

    pub(super) fn sampler_free(&mut self, operation: &Value) -> Result<Value> {
        exact_fields(operation, &["Op", "Sampler"])?;
        let sid = uint(operation, "Sampler")?;
        let removed = self
            .samplers
            .remove(&sid)
            .ok_or_else(|| invalid("unknown LLG sampler handle"))?;
        if let Sampler::Chain(children) = removed {
            for child in children {
                let _ = self.samplers.remove(&child);
            }
        }
        Ok(Value::Null)
    }
}

fn parse_indices(value: &Value) -> Result<Vec<usize>> {
    value
        .as_array()
        .ok_or_else(|| invalid("LLG Indices must be array"))?
        .iter()
        .map(|item| {
            item.as_u64()
                .ok_or_else(|| invalid("LLG probe index must be non-negative integer"))
                .and_then(|v| Ok(usize::try_from(v)?))
        })
        .collect()
}

/// Honest scope identity for `reference_llguidance`.
pub(super) fn identity() -> Value {
    json!({
        "Kind": "independent-rust-reference",
        "SupportedSourceIds": [ID],
        "UpstreamCommit": super::UPSTREAM,
        "AssertionSource": {"Path": ID, "Sha256": SOURCE_SHA256},
        "Arithmetic": "earley-utf8-cfg-and-bit-exact-candidates-v1",
        "NativeReferenceReplay": false,
        "ProductionRuntimeCalled": false,
        "CompleteSourceGate": false,
        "NativeRetirementQualified": false,
    })
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    use serde_json::json;

    use super::super::reference_vocab::load_gguf_vocab;
    use super::*;

    fn encode_gguf_str(out: &mut Vec<u8>, bytes: &[u8]) -> Result<()> {
        out.extend_from_slice(&u64::try_from(bytes.len())?.to_le_bytes());
        out.extend_from_slice(bytes);
        Ok(())
    }

    fn build_byte_bpe_vocab(path: &Path) -> Result<Arc<ReferenceVocab>> {
        let mut tokens: Vec<Vec<u8>> = Vec::new();
        // Token 0 = EOS
        tokens.push(b"<|eot_id|>".to_vec());
        // Single-byte tokens for all UTF-8 / ASCII scalars via direct identity on printable or byte fallback hex?
        // Actually use normal GPT-2 byte alphabet mapping for all 256 bytes!
        let mut next_extra = 256_u32;
        for b in 0..=u8::MAX {
            let scalar = if matches!(b, b'!'..=b'~' | 0xa1..=0xac | 0xae..=u8::MAX) {
                u32::from(b)
            } else {
                let s = next_extra;
                next_extra += 1;
                s
            };
            let ch = char::from_u32(scalar).ok_or_else(|| invalid("bad char"))?;
            tokens.push(ch.to_string().into_bytes());
        }
        // Add multi-char BPE tokens to exercise sub-word & multi-token trie traversal!
        for multi in [
            "true",
            "false",
            "null",
            "func(",
            "street_name",
            "ALLĠYOURĠBASE",
        ] {
            tokens.push(multi.as_bytes().to_vec());
        }
        let mut types: Vec<i32> = vec![3];
        types.resize(tokens.len(), 1);
        let mut kvs: Vec<(String, u32, Vec<u8>)> = Vec::new();
        {
            let mut b = Vec::new();
            encode_gguf_str(&mut b, b"gpt2")?;
            kvs.push(("tokenizer.ggml.model".into(), 8, b));
        }
        {
            let mut b = Vec::new();
            b.extend_from_slice(&8_u32.to_le_bytes());
            b.extend_from_slice(&u64::try_from(tokens.len())?.to_le_bytes());
            for tok in &tokens {
                encode_gguf_str(&mut b, tok)?;
            }
            kvs.push(("tokenizer.ggml.tokens".into(), 9, b));
        }
        {
            let mut b = Vec::new();
            b.extend_from_slice(&5_u32.to_le_bytes());
            b.extend_from_slice(&u64::try_from(types.len())?.to_le_bytes());
            for &ty in &types {
                b.extend_from_slice(&ty.to_le_bytes());
            }
            kvs.push(("tokenizer.ggml.token_type".into(), 9, b));
        }
        kvs.push((
            "tokenizer.ggml.eos_token_id".into(),
            4,
            0_u32.to_le_bytes().to_vec(),
        ));
        let mut raw = b"GGUF".to_vec();
        raw.extend_from_slice(&3_u32.to_le_bytes());
        raw.extend_from_slice(&0_u64.to_le_bytes());
        raw.extend_from_slice(&u64::try_from(kvs.len())?.to_le_bytes());
        for (k, kind, data) in kvs {
            encode_gguf_str(&mut raw, k.as_bytes())?;
            raw.extend_from_slice(&kind.to_le_bytes());
            raw.extend_from_slice(&data);
        }
        fs::write(path, raw)?;
        load_gguf_vocab(path)
    }

    fn candidate_bytes_vec(count: usize) -> Result<Vec<u8>> {
        let mut full = Vec::with_capacity(count * 12);
        for idx in 0..count {
            full.extend(i32::try_from(idx)?.to_le_bytes());
            full.extend(0_u32.to_le_bytes());
            full.extend(0_u32.to_le_bytes());
        }
        Ok(full)
    }

    fn check_input(
        reg: &mut LlgRegistry,
        vocab: &ReferenceVocab,
        tmpl_id: u64,
        sampler_id: u64,
        input: &str,
    ) -> Result<bool> {
        reg.sampler_reset(&json!({"Op":"llg-reset","Sampler":sampler_id}))?;
        let tokens = vocab.tokenize(input.as_bytes(), false, false)?;
        let cand_id = reg
            .candidates_clone(&json!({"Op":"llg-candidates-clone","Source":tmpl_id}))?["Candidates"]
            .as_u64()
            .ok_or_else(|| invalid("missing clone id"))?;
        for tok in tokens {
            reg.candidates_copy(&json!({"Op":"llg-candidates-copy","Candidates":cand_id,"Source":tmpl_id,"Encoding":"logits-le","Metadata":null}))?;
            let obs = reg.sampler_apply(&json!({"Op":"llg-apply","Sampler":sampler_id,"Candidates":cand_id,"Indices":[tok]}))?;
            let bits = u32::try_from(uint(field(&obs["Probes"][0], "Candidate")?, "LogitBits")?)?;
            if f32::from_bits(bits) < 0.0 {
                reg.candidates_free(&json!({"Op":"llg-candidates-free","Candidates":cand_id}))?;
                return Ok(false);
            }
            reg.sampler_accept(&json!({"Op":"llg-accept","Sampler":sampler_id,"Token":tok}))?;
        }
        reg.candidates_patch(&json!({"Op":"llg-candidates-patch","Candidates":cand_id,"Patches":[{"Index":0,"Fields":{"LogitBits":0}}]}))?;
        let obs = reg.sampler_apply(
            &json!({"Op":"llg-apply","Sampler":sampler_id,"Candidates":cand_id,"Indices":[0]}),
        )?;
        reg.candidates_free(&json!({"Op":"llg-candidates-free","Candidates":cand_id}))?;
        let eos_bits = u32::try_from(uint(field(&obs["Probes"][0], "Candidate")?, "LogitBits")?)?;
        Ok(f32::from_bits(eos_bits) >= 0.0)
    }

    #[test]
    fn independent_llg_executes_all_42_cases_344_predicates_and_chain() -> Result<()> {
        let stamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let dir =
            std::env::temp_dir().join(format!("prism-ref-llg-{}-{stamp}", std::process::id()));
        fs::create_dir(&dir)?;
        let vocab_path = dir.join("bpe.gguf");
        let vocab = build_byte_bpe_vocab(&vocab_path)?;
        let tmpl_path = dir.join("tmpl.bin");
        fs::write(&tmpl_path, candidate_bytes_vec(vocab.vocab_size())?)?;

        let mut reg = LlgRegistry::new();
        let info = LlgRegistry::vocabulary_info(&vocab);
        assert_eq!(info["Count"], vocab.vocab_size());
        let tmpl_id = reg.candidates_new(&json!({"Op":"llg-candidates-new","Count":vocab.vocab_size(),"File":tmpl_path,"Selected":-1,"Sorted":false}))?["Candidates"]
            .as_u64()
            .ok_or_else(|| invalid("missing tmpl id"))?;

        let fixture: Value = serde_json::from_str(include_str!("llguidance-cases.json"))?;
        let cases = fixture["Cases"]
            .as_array()
            .ok_or_else(|| invalid("missing Cases array"))?;
        assert_eq!(cases.len(), 42);
        let mut total_checks = 0_usize;
        for (idx, case) in cases.iter().enumerate() {
            let kind = case["Kind"].as_str().ok_or_else(|| invalid("kind"))?;
            let literal = case["Literal"].as_str().ok_or_else(|| invalid("literal"))?;
            let grammar = if kind == "schema" {
                format!("%llguidance {{}}\nstart: %json {literal}")
            } else {
                literal.to_owned()
            };
            let sampler_id = reg.sampler_new_grammar(
                &json!({"Op":"llg-new","Model":1,"GrammarKind":"lark","GrammarData":grammar}),
                &vocab,
            )?["Sampler"]
                .as_u64()
                .ok_or_else(|| invalid("missing sampler"))?;
            for (key, expected) in [("Passing", true), ("Failing", false)] {
                let list = case[key].as_array().ok_or_else(|| invalid("predicates"))?;
                for item in list {
                    let input = item.as_str().ok_or_else(|| invalid("text"))?;
                    let actual = check_input(&mut reg, &vocab, tmpl_id, sampler_id, input)?;
                    assert_eq!(
                        actual, expected,
                        "group {idx} ({}) failed on {key} input {input:?}",
                        case["Description"]
                    );
                    total_checks += 1;
                }
            }
            reg.sampler_free(&json!({"Op":"llg-free","Sampler":sampler_id}))?;
        }
        assert_eq!(total_checks, 344);

        // Verify sampler chain + dist deterministically selects patched logits over 100.0.
        let chain_id =
            reg.sampler_new_chain(&json!({"Op":"llg-chain-new","NoPerf":false}))?["Sampler"]
                .as_u64()
                .ok_or_else(|| invalid("chain"))?;
        let g_id = reg.sampler_new_grammar(
            &json!({"Op":"llg-new","Model":1,"GrammarKind":"lark","GrammarData":"%llguidance {}\nstart: /[A-Z ]*/"}),
            &vocab,
        )?["Sampler"]
            .as_u64()
            .ok_or_else(|| invalid("chain grammar"))?;
        let d_id = reg.sampler_new_dist(&json!({"Op":"llg-dist-new","Seed":42}))?["Sampler"]
            .as_u64()
            .ok_or_else(|| invalid("dist"))?;
        reg.sampler_chain_add(&json!({"Op":"llg-chain-add","Chain":chain_id,"Sampler":g_id}))?;
        reg.sampler_chain_add(&json!({"Op":"llg-chain-add","Chain":chain_id,"Sampler":d_id}))?;
        let tokens = vocab.tokenize(b"ALL YOUR BASE ARE BELONG TO US", false, false)?;
        let cand_id = reg
            .candidates_clone(&json!({"Op":"llg-candidates-clone","Source":tmpl_id}))?["Candidates"]
            .as_u64()
            .ok_or_else(|| invalid("chain cand"))?;
        for &tok in tokens.iter().chain(std::iter::once(&0)) {
            let count = vocab.vocab_size();
            reg.candidates_copy(&json!({"Op":"llg-candidates-copy","Candidates":cand_id,"Source":tmpl_id,"Encoding":"id-logit-le","Metadata":{"Offset":0,"Size":count,"Selected":-1,"Sorted":false}}))?;
            reg.candidates_patch(&json!({"Op":"llg-candidates-patch","Candidates":cand_id,"Patches":[{"Index":tok,"Fields":{"LogitBits":100.0_f32.to_bits()}}]}))?;
            let applied = reg.sampler_apply(
                &json!({"Op":"llg-apply","Sampler":chain_id,"Candidates":cand_id,"Indices":[]}),
            )?;
            assert_eq!(applied["SelectedBacking"]["Id"], tok);
            if tok != 0 {
                reg.sampler_accept(&json!({"Op":"llg-accept","Sampler":chain_id,"Token":tok}))?;
            }
        }
        reg.sampler_free(&json!({"Op":"llg-free","Sampler":chain_id}))?;
        reg.candidates_free(&json!({"Op":"llg-candidates-free","Candidates":cand_id}))?;
        reg.candidates_free(&json!({"Op":"llg-candidates-free","Candidates":tmpl_id}))?;
        fs::remove_dir_all(&dir)?;
        assert_eq!(identity()["NativeReferenceReplay"], false);
        Ok(())
    }

    #[test]
    fn independent_candidate_buffers_preserve_exact_nan_and_neg_zero_bits() -> Result<()> {
        let stamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let dir =
            std::env::temp_dir().join(format!("prism-ref-cand-{}-{stamp}", std::process::id()));
        fs::create_dir(&dir)?;
        let mut reg = LlgRegistry::new();
        let count = 257_usize;
        let patterns = [
            0_u32,
            0x8000_0000,
            0x7f80_0000,
            0xff80_0000,
            0x7fc1_2345,
            0x7fa1_2345,
            0x3f80_0000,
            0xbf80_0000,
        ];
        let mut source = Vec::new();
        let mut initial = Vec::new();
        let mut logits = Vec::new();
        let mut id_logits = Vec::new();
        for index in 0..count {
            let id = i32::try_from(count - index)?.to_le_bytes();
            let logit = patterns[index % patterns.len()].to_le_bytes();
            let prob = patterns[(index + 3) % patterns.len()].to_le_bytes();
            source.extend([id, logit, prob].concat());
            logits.extend(logit);
            id_logits.extend([id, logit].concat());
            initial.extend(
                [
                    i32::try_from(index)?.to_le_bytes(),
                    patterns[(index + 1) % patterns.len()].to_le_bytes(),
                    patterns[(index + 2) % patterns.len()].to_le_bytes(),
                ]
                .concat(),
            );
        }
        let source_file = dir.join("source.bin");
        let initial_file = dir.join("initial.bin");
        fs::write(&source_file, &source)?;
        fs::write(&initial_file, &initial)?;
        let tmpl = reg.candidates_new(&json!({"Op":"llg-candidates-new","Count":count,"File":source_file,"Selected":-1,"Sorted":false}))?["Candidates"]
            .as_u64()
            .ok_or_else(|| invalid("tmpl"))?;
        let indices: Vec<usize> = (0..count).collect();
        let untouched = reg.candidates_inspect(
            &json!({"Op":"llg-candidates-inspect","Candidates":tmpl,"Indices":indices}),
        )?;
        for (encoding, bytes) in [
            ("candidate-le", &source),
            ("id-logit-le", &id_logits),
            ("logits-le", &logits),
        ] {
            let file = dir.join(format!("{encoding}.bin"));
            fs::write(&file, bytes)?;
            let expected = reg.candidates_new(&json!({"Op":"llg-candidates-new","Count":count,"File":initial_file,"Selected":-1,"Sorted":false}))?["Candidates"]
                .as_u64()
                .ok_or_else(|| invalid("exp"))?;
            let actual = reg.candidates_new(&json!({"Op":"llg-candidates-new","Count":count,"File":initial_file,"Selected":-1,"Sorted":false}))?["Candidates"]
                .as_u64()
                .ok_or_else(|| invalid("act"))?;
            for cid in [expected, actual] {
                reg.candidates_load(&json!({"Op":"llg-candidates-load","Candidates":cid,"File":initial_file,"Encoding":"candidate-le","Metadata":{"Offset":2,"Size":count-4,"Selected":5,"Sorted":true}}))?;
            }
            reg.candidates_load(&json!({"Op":"llg-candidates-load","Candidates":expected,"File":file,"Encoding":encoding,"Metadata":null}))?;
            reg.candidates_copy(&json!({"Op":"llg-candidates-copy","Candidates":actual,"Source":tmpl,"Encoding":encoding,"Metadata":null}))?;
            let obs = reg.candidates_inspect(
                &json!({"Op":"llg-candidates-inspect","Candidates":actual,"Indices":indices}),
            )?;
            assert_eq!(
                obs,
                reg.candidates_inspect(
                    &json!({"Op":"llg-candidates-inspect","Candidates":expected,"Indices":indices})
                )?
            );
            assert_eq!(
                reg.candidates_inspect(
                    &json!({"Op":"llg-candidates-inspect","Candidates":tmpl,"Indices":indices})
                )?,
                untouched
            );
            assert_eq!(
                obs["Probes"][5]["Candidate"]["LogitBits"],
                patterns[5 % patterns.len()]
            );
            reg.candidates_free(&json!({"Op":"llg-candidates-free","Candidates":expected}))?;
            reg.candidates_free(&json!({"Op":"llg-candidates-free","Candidates":actual}))?;
        }
        reg.candidates_patch(&json!({"Op":"llg-candidates-patch","Candidates":tmpl,"Patches":[{"Index":7,"Fields":{"LogitBits":0x7fc0_dead_u32}}]}))?;
        let patched = reg.candidates_inspect(
            &json!({"Op":"llg-candidates-inspect","Candidates":tmpl,"Indices":[7]}),
        )?;
        assert_eq!(
            patched["Probes"][0]["Candidate"]["LogitBits"],
            0x7fc0_dead_u32
        );
        reg.candidates_free(&json!({"Op":"llg-candidates-free","Candidates":tmpl}))?;
        fs::remove_dir_all(&dir)?;
        Ok(())
    }
}
