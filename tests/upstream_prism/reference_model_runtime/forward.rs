//! Clean-room safe Rust transformer, `DSpark`, `KV` cache, mean-centering, and trace execution.
//!
//! No `bonsai_prism_runtime`, `bonsai_gguf`, or native GGML libraries are linked or called.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::sync::Arc;

use serde_json::{Value, json};

use super::gguf_model::{GgufValue, ModelData, checked_session_file, f64_to_f32};
use crate::{Result, invalid};

const MAX_CONTEXTS: usize = 64;
const MAX_BATCHES: usize = 64;
const MAX_TRACES: usize = 64;

#[derive(Clone, Debug)]
pub(super) struct BatchRow {
    pub(super) token: i32,
    pub(super) pos: i32,
    pub(super) seq_ids: Vec<i32>,
    pub(super) logits: bool,
}

#[derive(Clone)]
pub(super) struct BatchData {
    pub(super) capacity: usize,
    pub(super) rows: Vec<BatchRow>,
}

impl BatchData {
    pub(super) const fn new(capacity: usize, _n_seq_max: usize) -> Self {
        Self {
            capacity,
            rows: Vec::new(),
        }
    }
}

#[derive(Clone, Default)]
pub(super) struct TraceCollector {
    pub(super) exact_names: BTreeSet<String>,
    pub(super) prefixes: Vec<String>,
    pub(super) values: BTreeMap<String, Vec<f32>>,
    pub(super) failed: bool,
}

impl TraceCollector {
    pub(super) fn wants(&self, name: &str) -> bool {
        self.exact_names.contains(name) || self.prefixes.iter().any(|p| name.starts_with(p))
    }

    pub(super) fn record(&mut self, name: &str, data: &[f32]) {
        if self.wants(name) {
            self.values.insert(name.to_owned(), data.to_vec());
        }
    }
}

#[derive(Clone)]
pub(super) struct KvEntry {
    pub(super) pos: i32,
    pub(super) k_layers: Vec<Vec<f32>>,
    pub(super) v_layers: Vec<Vec<f32>>,
    pub(super) hidden: Vec<f32>,
    pub(super) token: i32,
}

#[derive(Clone, Default)]
pub(super) struct SeqMemory {
    pub(super) entries: Vec<KvEntry>,
    pub(super) recurrent_state: Vec<f32>,
    pub(super) rollback_snapshots: BTreeMap<i32, (Vec<KvEntry>, Vec<f32>)>,
    pub(super) checkpoints: BTreeMap<i32, (Vec<KvEntry>, Vec<f32>)>,
}

#[derive(Clone, Default)]
pub(super) struct RuntimeEnv {
    pub(super) vars: BTreeMap<String, Value>,
}

impl RuntimeEnv {
    pub(super) fn apply(&mut self, changes: &Value) -> Result<()> {
        let obj = changes
            .as_object()
            .ok_or_else(|| invalid("process-environment Changes must be object"))?;
        for (k, v) in obj {
            if v.is_null() {
                self.vars.remove(k);
            } else {
                self.vars.insert(k.clone(), v.clone());
            }
        }
        Ok(())
    }

    pub(super) fn get_bool(&self, key: &str) -> bool {
        match self.vars.get(key) {
            Some(Value::Bool(b)) => *b,
            Some(Value::String(s)) => matches!(s.as_str(), "1" | "true" | "TRUE" | "yes"),
            Some(Value::Number(n)) => n.as_i64().is_some_and(|v| v != 0),
            _ => false,
        }
    }

    pub(super) fn get_raw_prefix(&self) -> Option<&Value> {
        self.vars.get("LLAMA_DSPARK_CORRECTION_PREFIX")
    }

    pub(super) fn parse_correction_prefix(&self, block_size: usize) -> Result<Option<usize>> {
        let Some(raw) = self.get_raw_prefix() else {
            return Ok(None);
        };
        let n_i64 = match raw {
            Value::Number(n) => n.as_i64(),
            Value::String(s) => s.parse::<i64>().ok(),
            _ => None,
        }
        .ok_or_else(|| invalid("runtime-error:invalid LLAMA_DSPARK_CORRECTION_PREFIX"))?;
        let n = usize::try_from(n_i64)
            .ok()
            .filter(|&v| v >= 1 && v <= block_size)
            .ok_or_else(|| {
                invalid("runtime-error:LLAMA_DSPARK_CORRECTION_PREFIX out of block range")
            })?;
        Ok(Some(n))
    }
}

#[derive(Clone)]
pub(super) struct DsparkStaged {
    pub(super) width: usize,
    pub(super) features: Vec<f32>,
    pub(super) positions: Vec<i32>,
}

#[derive(Clone)]
pub(super) struct ContextState {
    pub(super) model: Arc<ModelData>,
    pub(super) n_ctx: usize,
    pub(super) n_batch: usize,
    pub(super) n_ubatch: usize,
    pub(super) n_seq_max: usize,
    pub(super) type_k: i32,
    pub(super) causal_attention: bool,
    pub(super) embeddings_only: bool,
    pub(super) trace_id: Option<u64>,
    pub(super) lora_adapters: Vec<(Arc<ModelData>, f32)>,
    pub(super) kv_bias: Option<Vec<Vec<f32>>>,
    pub(super) capture_layers: Vec<i32>,
    pub(super) captured_rows: Vec<Vec<Vec<f32>>>,
    pub(super) staged: Option<DsparkStaged>,
    pub(super) sequences: BTreeMap<i32, SeqMemory>,
    pub(super) output_logits: Vec<Vec<f32>>,
    pub(super) output_embeddings: Vec<Vec<f32>>,
    pub(super) sampled_tokens: Vec<i32>,
    pub(super) seq_samplers: BTreeMap<i32, u64>,
    pub(super) aborted: bool,
}

impl ContextState {
    pub(super) fn kv_seq_min(&self, seq: i32) -> i32 {
        self.sequences
            .get(&seq)
            .and_then(|s| s.entries.iter().map(|e| e.pos).min())
            .unwrap_or(-1)
    }

    pub(super) fn kv_seq_max(&self, seq: i32) -> i32 {
        self.sequences
            .get(&seq)
            .and_then(|s| s.entries.iter().map(|e| e.pos).max())
            .unwrap_or(-1)
    }

    pub(super) fn memory_clear(&mut self) {
        self.sequences.clear();
        self.output_logits.clear();
        self.output_embeddings.clear();
        self.sampled_tokens.clear();
        self.captured_rows.clear();
    }

    pub(super) fn memory_remove(&mut self, seq: i32, p0: i32, p1: i32) -> bool {
        let is_rec = self.model.is_recurrent || self.model.is_hybrid;
        if seq < 0 {
            self.sequences.clear();
            return true;
        }
        let Some(mem) = self.sequences.get_mut(&seq) else {
            return true;
        };
        if !is_rec {
            mem.entries
                .retain(|e| !((p0 < 0 || e.pos >= p0) && (p1 < 0 || e.pos < p1)));
            return true;
        }
        if p0 <= 0 && p1 < 0 {
            mem.entries.clear();
            mem.recurrent_state.fill(0.0);
            mem.rollback_snapshots.clear();
            return true;
        }
        if p1 >= 0 && mem.entries.iter().any(|e| e.pos >= p1) {
            return false;
        }
        if let Some((saved_entries, saved_rec)) = mem.rollback_snapshots.remove(&p0) {
            mem.entries = saved_entries;
            mem.recurrent_state = saved_rec;
            mem.rollback_snapshots.clear();
            return true;
        }
        false
    }

    pub(super) fn memory_cp(&mut self, src: i32, dst: i32, p0: i32, p1: i32) {
        let copied = self.sequences.get(&src).cloned().unwrap_or_default();
        let filtered: Vec<KvEntry> = copied
            .entries
            .into_iter()
            .filter(|e| (p0 < 0 || e.pos >= p0) && (p1 < 0 || e.pos < p1))
            .collect();
        self.sequences.insert(
            dst,
            SeqMemory {
                entries: filtered,
                recurrent_state: copied.recurrent_state,
                rollback_snapshots: copied.rollback_snapshots,
                checkpoints: BTreeMap::new(),
            },
        );
    }

    pub(super) fn memory_keep(&mut self, seq: i32) {
        self.sequences.retain(|&id, _| id == seq);
    }

    pub(super) fn memory_add(&mut self, seq: i32, p0: i32, p1: i32, delta: i32) {
        if let Some(mem) = self.sequences.get_mut(&seq) {
            for e in &mut mem.entries {
                if (p0 < 0 || e.pos >= p0) && (p1 < 0 || e.pos < p1) {
                    e.pos = e.pos.saturating_add(delta);
                }
            }
        }
    }

    pub(super) fn checkpoint_update(&mut self, slot: i32, seq: i32) {
        let mem = self.sequences.entry(seq).or_default();
        let snap = (mem.entries.clone(), mem.recurrent_state.clone());
        mem.checkpoints.insert(slot, snap);
    }

    pub(super) fn checkpoint_load(&mut self, slot: i32, seq: i32) -> Result<()> {
        let mem = self
            .sequences
            .get_mut(&seq)
            .ok_or_else(|| invalid("unknown sequence in checkpoint load"))?;
        let (entries, rec) = mem
            .checkpoints
            .get(&slot)
            .cloned()
            .ok_or_else(|| invalid("checkpoint slot not found"))?;
        mem.entries = entries;
        mem.recurrent_state = rec;
        mem.rollback_snapshots.clear();
        Ok(())
    }

    pub(super) fn encode_state_bytes(&self, seq: i32) -> Result<Vec<u8>> {
        let mem = self.sequences.get(&seq);
        let mut out = Vec::from(*b"PRSMST01");
        let entries = mem.map_or(&[][..], |m| m.entries.as_slice());
        out.extend_from_slice(&u32::try_from(entries.len())?.to_le_bytes());
        for entry in entries {
            write_entry_bytes(&mut out, entry)?;
        }
        let rec = mem.map_or(&[][..], |m| m.recurrent_state.as_slice());
        out.extend_from_slice(&u32::try_from(rec.len())?.to_le_bytes());
        for &v in rec {
            out.extend_from_slice(&v.to_le_bytes());
        }
        let n_logits = u32::try_from(self.output_logits.len())?;
        out.extend_from_slice(&n_logits.to_le_bytes());
        for row in &self.output_logits {
            out.extend_from_slice(&u32::try_from(row.len())?.to_le_bytes());
            for &v in row {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        Ok(out)
    }

    pub(super) fn decode_state_bytes(&mut self, seq: i32, data: &[u8]) -> Result<()> {
        let mut cursor = std::io::Cursor::new(data);
        let mut magic = [0_u8; 8];
        std::io::Read::read_exact(&mut cursor, &mut magic)?;
        if &magic != b"PRSMST01" {
            return Err(invalid("invalid serialized state blob"));
        }
        let n_entries = usize::try_from(read_u32(&mut cursor)?)?;
        let mut entries = Vec::with_capacity(n_entries);
        for _ in 0..n_entries {
            entries.push(read_entry_bytes(&mut cursor)?);
        }
        let n_rec = usize::try_from(read_u32(&mut cursor)?)?;
        let mut recurrent_state = Vec::with_capacity(n_rec);
        for _ in 0..n_rec {
            recurrent_state.push(read_f32(&mut cursor)?);
        }
        let n_logits = usize::try_from(read_u32(&mut cursor)?)?;
        let mut logits = Vec::with_capacity(n_logits);
        for _ in 0..n_logits {
            let n = usize::try_from(read_u32(&mut cursor)?)?;
            let mut row = Vec::with_capacity(n);
            for _ in 0..n {
                row.push(read_f32(&mut cursor)?);
            }
            logits.push(row);
        }
        self.sequences.insert(
            seq,
            SeqMemory {
                entries,
                recurrent_state,
                rollback_snapshots: BTreeMap::new(),
                checkpoints: BTreeMap::new(),
            },
        );
        if !logits.is_empty() {
            self.output_logits = logits;
        }
        Ok(())
    }
}

fn read_u32(r: &mut std::io::Cursor<&[u8]>) -> Result<u32> {
    let mut b = [0_u8; 4];
    std::io::Read::read_exact(r, &mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn read_f32(r: &mut std::io::Cursor<&[u8]>) -> Result<f32> {
    Ok(f32::from_bits(read_u32(r)?))
}

fn write_entry_bytes(out: &mut Vec<u8>, entry: &KvEntry) -> Result<()> {
    out.extend_from_slice(&entry.pos.to_le_bytes());
    out.extend_from_slice(&entry.token.to_le_bytes());
    out.extend_from_slice(&u32::try_from(entry.hidden.len())?.to_le_bytes());
    for &v in &entry.hidden {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out.extend_from_slice(&u32::try_from(entry.k_layers.len())?.to_le_bytes());
    for (k, v) in entry.k_layers.iter().zip(&entry.v_layers) {
        out.extend_from_slice(&u32::try_from(k.len())?.to_le_bytes());
        for &val in k {
            out.extend_from_slice(&val.to_le_bytes());
        }
        for &val in v {
            out.extend_from_slice(&val.to_le_bytes());
        }
    }
    Ok(())
}

fn read_entry_bytes(r: &mut std::io::Cursor<&[u8]>) -> Result<KvEntry> {
    let pos = i32::from_le_bytes(read_u32(r)?.to_le_bytes());
    let token = i32::from_le_bytes(read_u32(r)?.to_le_bytes());
    let n_hid = usize::try_from(read_u32(r)?)?;
    let mut hidden = Vec::with_capacity(n_hid);
    for _ in 0..n_hid {
        hidden.push(read_f32(r)?);
    }
    let n_layers = usize::try_from(read_u32(r)?)?;
    let mut k_layers = Vec::with_capacity(n_layers);
    let mut v_layers = Vec::with_capacity(n_layers);
    for _ in 0..n_layers {
        let len = usize::try_from(read_u32(r)?)?;
        let mut k = Vec::with_capacity(len);
        let mut v = Vec::with_capacity(len);
        for _ in 0..len {
            k.push(read_f32(r)?);
        }
        for _ in 0..len {
            v.push(read_f32(r)?);
        }
        k_layers.push(k);
        v_layers.push(v);
    }
    Ok(KvEntry {
        pos,
        k_layers,
        v_layers,
        hidden,
        token,
    })
}

pub(super) fn load_bias_gguf(
    path: &Path,
    expected_layers: usize,
    require_rotated: bool,
) -> Result<Option<Vec<Vec<f32>>>> {
    let model = ModelData::load_from_file(path, true, None, None)?
        .ok_or_else(|| invalid("bias GGUF cancelled"))?;
    let is_rot = model
        .kvs
        .get("kv_mean_center.rotated")
        .and_then(GgufValue::as_bool_coerce)
        .unwrap_or(false);
    if require_rotated && !is_rot {
        return Ok(None);
    }
    let mut layers = Vec::with_capacity(expected_layers);
    for idx in 0..expected_layers {
        let name = format!("blk.{idx}.attn_k_bias");
        layers.push(model.tensor_f32(&name)?);
    }
    Ok(Some(layers))
}

pub(super) fn check_context_limits(contexts: &BTreeMap<u64, ContextState>) -> Result<()> {
    if contexts.len() >= MAX_CONTEXTS {
        return Err(invalid("context handle bound exceeded"));
    }
    Ok(())
}

pub(super) fn check_batch_limits(batches: &BTreeMap<u64, BatchData>) -> Result<()> {
    if batches.len() >= MAX_BATCHES {
        return Err(invalid("batch handle bound exceeded"));
    }
    Ok(())
}

pub(super) fn check_trace_limits(traces: &BTreeMap<u64, TraceCollector>) -> Result<()> {
    if traces.len() >= MAX_TRACES {
        return Err(invalid("trace handle bound exceeded"));
    }
    Ok(())
}

pub(super) fn build_context(
    model: Arc<ModelData>,
    operation: &Value,
    session_dir: &Path,
    params_ctx: Option<&super::params::SessionParams>,
) -> Result<Option<ContextState>> {
    let mut n_ctx = params_ctx.map_or(model.n_ctx_train, |p| {
        usize::try_from(p.n_ctx).unwrap_or(512)
    });
    if n_ctx == 0 {
        n_ctx = model.n_ctx_train.max(512);
    }
    let mut n_batch = params_ctx.map_or(512, |p| usize::try_from(p.n_batch).unwrap_or(512));
    let mut micro_batch = params_ctx.map_or(512, |p| usize::try_from(p.n_ubatch).unwrap_or(512));
    let mut n_seq_max = params_ctx.map_or(1, |p| usize::try_from(p.n_parallel).unwrap_or(1));
    let mut type_k = 1_i32;
    let mut embeddings_only = false;
    let mut trace_id = None;
    let mut mean_center_path: Option<String> = None;

    if let Some(ov) = operation.get("Overrides").and_then(Value::as_object) {
        if let Some(v) = ov.get("Context").and_then(Value::as_u64) {
            n_ctx = if v == 0 {
                model.n_ctx_train.max(512)
            } else {
                usize::try_from(v)?
            };
        }
        if let Some(v) = ov.get("Batch").and_then(Value::as_u64) {
            n_batch = usize::try_from(v)?;
        }
        if let Some(v) = ov.get("MicroBatch").and_then(Value::as_u64) {
            micro_batch = usize::try_from(v)?;
        }
        if let Some(v) = ov.get("Sequences").and_then(Value::as_u64) {
            n_seq_max = usize::try_from(v)?;
        }
        if let Some(v) = ov.get("TypeK").and_then(Value::as_i64) {
            type_k = i32::try_from(v)?;
        }
        if let Some(v) = ov.get("Embeddings").and_then(Value::as_bool) {
            embeddings_only = v;
        }
        if let Some(v) = ov.get("TraceId").and_then(Value::as_u64) {
            trace_id = Some(v);
        }
        if let Some(v) = ov.get("MeanCenterPath").and_then(Value::as_str) {
            mean_center_path = Some(v.to_owned());
        }
    }

    let kv_bias = if let Some(raw_path) = mean_center_path {
        if type_k != 2 {
            return Ok(None);
        }
        let full = checked_session_file(&raw_path, session_dir, true)?;
        let Some(layers) = load_bias_gguf(&full, model.n_layer, true)? else {
            return Ok(None);
        };
        Some(layers)
    } else {
        None
    };

    Ok(Some(ContextState {
        model,
        n_ctx: n_ctx.max(16),
        n_batch: n_batch.max(16),
        n_ubatch: micro_batch.max(16),
        n_seq_max: n_seq_max.max(1),
        type_k,
        causal_attention: true,
        embeddings_only,
        trace_id,
        lora_adapters: Vec::new(),
        kv_bias,
        capture_layers: Vec::new(),
        captured_rows: Vec::new(),
        staged: None,
        sequences: BTreeMap::new(),
        output_logits: Vec::new(),
        output_embeddings: Vec::new(),
        sampled_tokens: Vec::new(),
        seq_samplers: BTreeMap::new(),
        aborted: false,
    }))
}

pub(super) fn run_decode(
    ctx: &mut ContextState,
    batch: &BatchData,
    traces: &mut BTreeMap<u64, TraceCollector>,
    env: &RuntimeEnv,
) -> Result<()> {
    if ctx.aborted {
        return Err(invalid("runtime-error:decode aborted by callback"));
    }
    if ctx.model.is_dspark {
        run_dspark_decode(ctx, batch, traces, env)
    } else {
        run_standard_decode(ctx, batch)
    }
}

fn matvec_f64_round(weight: &[f32], input: &[f32]) -> Result<Vec<f32>> {
    if input.is_empty() || !weight.len().is_multiple_of(input.len()) {
        return Err(invalid("matrix-vector shape mismatch"));
    }
    Ok(weight
        .chunks_exact(input.len())
        .map(|row| {
            let sum: f64 = row
                .iter()
                .zip(input)
                .map(|(&w, &x)| f64::from(w) * f64::from(x))
                .sum();
            f64_to_f32(sum)
        })
        .collect())
}

fn rms_norm(input: &[f32], weight: Option<&[f32]>, eps: f32) -> Result<Vec<f32>> {
    if input.is_empty() {
        return Ok(Vec::new());
    }
    let sq: f64 = input.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
    let denom = f64_to_f32(sq / f64::from(u32::try_from(input.len())?)) + eps;
    let scale = denom.sqrt().recip();
    Ok(match weight {
        Some(w) if w.len() == input.len() => input
            .iter()
            .zip(w)
            .map(|(&v, &wt)| v * (scale * wt))
            .collect(),
        _ => input.iter().map(|&v| v * scale).collect(),
    })
}

fn tensor_slice(model: &ModelData, name: &str) -> Result<Option<Vec<f32>>> {
    if model.tensors.contains_key(name) {
        Ok(Some(model.tensor_f32(name)?))
    } else {
        Ok(None)
    }
}

fn embed_token(model: &ModelData, token: i32) -> Result<Vec<f32>> {
    let embd = model.n_embd;
    let tok_u = usize::try_from(token.max(0)).unwrap_or(0);
    if model.tensors.contains_key("token_embd.weight") {
        let all = model.tensor_f32("token_embd.weight")?;
        if let Some(slice) = all.get(tok_u * embd..(tok_u + 1) * embd) {
            return Ok(slice.to_vec());
        }
    }
    let mut vec = vec![0.0_f32; embd];
    for (i, slot) in vec.iter_mut().enumerate() {
        let angle = f64::from(u32::try_from((tok_u + 1) * (i + 1))?) * 0.13;
        *slot = f64_to_f32(angle.sin() * 0.25);
    }
    Ok(vec)
}

fn apply_rope(vec: &mut [f32], pos: i32, head_dim: usize, freq_base: f32) {
    if head_dim < 2 || !vec.len().is_multiple_of(head_dim) {
        return;
    }
    let pos_f = f64::from(pos);
    let base_f = f64::from(freq_base);
    for head in vec.chunks_exact_mut(head_dim) {
        for pair in 0..head_dim / 2 {
            let exp = -f64::from(u32::try_from(pair * 2).unwrap_or(0))
                / f64::from(u32::try_from(head_dim).unwrap_or(2));
            let theta = pos_f * base_f.powf(exp);
            let c = theta.cos();
            let s = theta.sin();
            let x0 = f64::from(head[2 * pair]);
            let x1 = f64::from(head[2 * pair + 1]);
            head[2 * pair] = f64_to_f32(x1.mul_add(-s, x0 * c));
            head[2 * pair + 1] = f64_to_f32(x1.mul_add(c, x0 * s));
        }
    }
}

fn linear_with_lora(
    model: &ModelData,
    weight_name: &str,
    input: &[f32],
    out_dim: usize,
    loras: &[(Arc<ModelData>, f32)],
) -> Result<Vec<f32>> {
    let mut out = if let Some(w) = tensor_slice(model, weight_name)? {
        matvec_f64_round(&w, input)?
    } else {
        vec![0.0_f32; out_dim]
    };
    let stem = weight_name.strip_suffix(".weight").unwrap_or(weight_name);
    for (adapter, user_scale) in loras {
        let a_name = format!("{stem}.lora_a");
        let b_name = format!("{stem}.lora_b");
        if let (Some(a_entry), Some(b_entry)) =
            (adapter.tensors.get(&a_name), adapter.tensors.get(&b_name))
        {
            let rank = usize::try_from(a_entry.ne[1].max(1))?;
            let alpha = adapter
                .kvs
                .get("general.lora.alpha")
                .and_then(GgufValue::as_f32_coerce)
                .unwrap_or(1.0);
            let scale = f64::from(user_scale * alpha) / f64::from(u32::try_from(rank)?);
            let a_vals = a_entry.to_f32()?;
            let b_vals = b_entry.to_f32()?;
            let hidden = matvec_f64_round(&a_vals, input)?;
            let delta = matvec_f64_round(&b_vals, &hidden)?;
            for (dst, d) in out.iter_mut().zip(delta) {
                *dst = f64_to_f32(scale.mul_add(f64::from(d), f64::from(*dst)));
            }
        }
    }
    Ok(out)
}

#[expect(
    clippy::too_many_lines,
    reason = "single-pass transformer decoder with KV cache, bias centering, capture, and recurrent rollback"
)]
fn run_standard_decode(ctx: &mut ContextState, batch: &BatchData) -> Result<()> {
    ctx.output_logits.clear();
    ctx.output_embeddings.clear();
    ctx.sampled_tokens.clear();
    ctx.captured_rows.clear();

    let model = Arc::clone(&ctx.model);
    let embd = model.n_embd;
    let n_layer = model.n_layer.min(32);
    let head_dim = (embd / model.n_head.max(1)).max(2);
    let kv_dim = head_dim * model.n_head_kv.max(1);
    let is_rec = model.is_recurrent || model.is_hybrid;

    if is_rec {
        let mut seen = BTreeSet::new();
        for row in &batch.rows {
            let seq_id = *row.seq_ids.first().unwrap_or(&0);
            if seen.insert(seq_id)
                && let Some(mem) = ctx.sequences.get_mut(&seq_id)
            {
                mem.rollback_snapshots.clear();
            }
        }
    }

    for row in &batch.rows {
        let seq_id = *row.seq_ids.first().unwrap_or(&0);
        if is_rec {
            let mem = ctx.sequences.entry(seq_id).or_default();
            let snap = (mem.entries.clone(), mem.recurrent_state.clone());
            mem.rollback_snapshots.insert(row.pos, snap);
        }
        let mut h = embed_token(&model, row.token)?;
        let mut k_layers = Vec::with_capacity(n_layer);
        let mut v_layers = Vec::with_capacity(n_layer);
        let mut captures_for_row = Vec::new();

        for layer in 0..n_layer {
            let anorm_w = tensor_slice(&model, &format!("blk.{layer}.attn_norm.weight"))?;
            let x_norm = rms_norm(&h, anorm_w.as_deref(), model.rms_eps)?;
            let mut q = linear_with_lora(
                &model,
                &format!("blk.{layer}.attn_q.weight"),
                &x_norm,
                embd,
                &ctx.lora_adapters,
            )?;
            let mut k = linear_with_lora(
                &model,
                &format!("blk.{layer}.attn_k.weight"),
                &x_norm,
                kv_dim.min(embd),
                &ctx.lora_adapters,
            )?;
            let v = linear_with_lora(
                &model,
                &format!("blk.{layer}.attn_v.weight"),
                &x_norm,
                kv_dim.min(embd),
                &ctx.lora_adapters,
            )?;
            apply_rope(&mut q, row.pos, head_dim, model.rope_freq_base);
            apply_rope(&mut k, row.pos, head_dim, model.rope_freq_base);
            if let Some(bias_layers) = &ctx.kv_bias
                && let Some(bias) = bias_layers.get(layer)
                && bias.len() == k.len()
            {
                for (ki, &bi) in k.iter_mut().zip(bias) {
                    *ki -= bi;
                }
            }
            if k_layers.len() <= layer {
                k_layers.push(k.clone());
                v_layers.push(v.clone());
            }
            let seq_mem = ctx.sequences.entry(seq_id).or_default();
            let attn_out = attend_layer(
                (&q, &k, &v),
                seq_mem,
                layer,
                row.pos,
                ctx.causal_attention,
                &model,
            )?;
            let o_proj = linear_with_lora(
                &model,
                &format!("blk.{layer}.attn_output.weight"),
                &attn_out,
                embd,
                &ctx.lora_adapters,
            )?;
            for (hi, oi) in h.iter_mut().zip(o_proj) {
                *hi += oi;
            }
            let fnorm_w = tensor_slice(&model, &format!("blk.{layer}.ffn_norm.weight"))?;
            let f_in = rms_norm(&h, fnorm_w.as_deref(), model.rms_eps)?;
            let gate = linear_with_lora(
                &model,
                &format!("blk.{layer}.ffn_gate.weight"),
                &f_in,
                model.n_ff,
                &ctx.lora_adapters,
            )?;
            let up = linear_with_lora(
                &model,
                &format!("blk.{layer}.ffn_up.weight"),
                &f_in,
                model.n_ff,
                &ctx.lora_adapters,
            )?;
            let gated: Vec<f32> = gate
                .into_iter()
                .zip(up)
                .map(|(g, u)| (g / (1.0 + (-g).exp())) * u)
                .collect();
            let down = linear_with_lora(
                &model,
                &format!("blk.{layer}.ffn_down.weight"),
                &gated,
                embd,
                &ctx.lora_adapters,
            )?;
            for (hi, di) in h.iter_mut().zip(down) {
                *hi += di;
            }
            if is_rec {
                seq_mem.recurrent_state.resize(embd, 0.0);
                for (rs, &hv) in seq_mem.recurrent_state.iter_mut().zip(&h) {
                    *rs = hv.mul_add(0.1, *rs * 0.9);
                }
            }
            if ctx.capture_layers.contains(&i32::try_from(layer)?) {
                captures_for_row.push(h.clone());
            }
        }

        if !captures_for_row.is_empty() {
            ctx.captured_rows.push(captures_for_row);
        }

        let seq_mem = ctx.sequences.entry(seq_id).or_default();
        seq_mem.entries.push(KvEntry {
            pos: row.pos,
            k_layers,
            v_layers,
            hidden: h.clone(),
            token: row.token,
        });

        if row.logits {
            let norm_w = tensor_slice(&model, "output_norm.weight")?;
            let final_h = rms_norm(&h, norm_w.as_deref(), model.rms_eps)?;
            if ctx.embeddings_only {
                ctx.output_embeddings.push(final_h);
            } else {
                let logits = project_logits(&model, &final_h, &ctx.lora_adapters)?;
                ctx.output_logits.push(logits);
            }
        }
    }
    Ok(())
}

fn attend_layer(
    qkv: (&[f32], &[f32], &[f32]),
    seq_mem: &SeqMemory,
    layer: usize,
    pos: i32,
    causal: bool,
    model: &ModelData,
) -> Result<Vec<f32>> {
    let (q, k_cur, v_cur) = qkv;
    let n_head = model.n_head.max(1);
    let n_head_kv = model.n_head_kv.max(1);
    let head_dim = (q.len() / n_head).max(1);
    let kv_head_dim = (k_cur.len() / n_head_kv).max(1);
    let scale = (f64::from(u32::try_from(head_dim)?)).sqrt().recip();
    let mut keys: Vec<(&[f32], &[f32])> = Vec::new();
    for entry in &seq_mem.entries {
        if (!causal || entry.pos <= pos)
            && let (Some(k_e), Some(v_e)) = (entry.k_layers.get(layer), entry.v_layers.get(layer))
        {
            keys.push((k_e.as_slice(), v_e.as_slice()));
        }
    }
    keys.push((k_cur, v_cur));
    let mut out = vec![0.0_f32; q.len()];
    for h_idx in 0..n_head {
        let kv_idx = (h_idx * n_head_kv) / n_head;
        let q_h = &q[h_idx * head_dim..(h_idx + 1) * head_dim];
        let mut scores = Vec::with_capacity(keys.len());
        let mut max_s = f64::NEG_INFINITY;
        for (k_all, _) in &keys {
            let k_h = &k_all[kv_idx * kv_head_dim..(kv_idx + 1) * kv_head_dim];
            let dot: f64 = q_h
                .iter()
                .zip(k_h)
                .map(|(&qi, &ki)| f64::from(qi) * f64::from(ki))
                .sum();
            let s = dot * scale;
            if s > max_s {
                max_s = s;
            }
            scores.push(s);
        }
        let exps: Vec<f64> = scores.into_iter().map(|s| (s - max_s).exp()).collect();
        let sum_exp: f64 = exps.iter().sum();
        let mut acc = vec![0.0_f64; head_dim];
        for (w, (_, v_all)) in exps.into_iter().zip(&keys) {
            let prob = w / sum_exp;
            let v_h = &v_all[kv_idx * kv_head_dim..(kv_idx + 1) * kv_head_dim];
            for (a, &vi) in acc.iter_mut().zip(v_h) {
                *a = prob.mul_add(f64::from(vi), *a);
            }
        }
        for (dst, val) in out[h_idx * head_dim..(h_idx + 1) * head_dim]
            .iter_mut()
            .zip(acc)
        {
            *dst = f64_to_f32(val);
        }
    }
    Ok(out)
}

fn project_logits(
    model: &ModelData,
    final_h: &[f32],
    loras: &[(Arc<ModelData>, f32)],
) -> Result<Vec<f32>> {
    if !model.output_buffer.is_empty() && model.output_buffer.len().is_multiple_of(final_h.len()) {
        let mut out = matvec_f64_round(&model.output_buffer, final_h)?;
        for (adapter, user_scale) in loras {
            let stem = if model.tensors.contains_key("output.weight") {
                "output"
            } else {
                "token_embd"
            };
            let a_name = format!("{stem}.lora_a");
            let b_name = format!("{stem}.lora_b");
            if let (Some(a_entry), Some(b_entry)) =
                (adapter.tensors.get(&a_name), adapter.tensors.get(&b_name))
            {
                let rank = usize::try_from(a_entry.ne[1].max(1))?;
                let alpha = adapter
                    .kvs
                    .get("general.lora.alpha")
                    .and_then(GgufValue::as_f32_coerce)
                    .unwrap_or(1.0);
                let scale = f64::from(user_scale * alpha) / f64::from(u32::try_from(rank)?);
                let hidden = matvec_f64_round(&a_entry.to_f32()?, final_h)?;
                let delta = matvec_f64_round(&b_entry.to_f32()?, &hidden)?;
                for (dst, d) in out.iter_mut().zip(delta) {
                    *dst = f64_to_f32(scale.mul_add(f64::from(d), f64::from(*dst)));
                }
            }
        }
        return Ok(out);
    }
    Ok(vec![0.0_f32; model.vocab_size])
}

#[expect(
    clippy::too_many_lines,
    reason = "DSpark dual-trunk forward pass with exact SNR/balanced traces, prefix gate, and Markov correction head"
)]
fn run_dspark_decode(
    ctx: &mut ContextState,
    batch: &BatchData,
    traces: &mut BTreeMap<u64, TraceCollector>,
    env: &RuntimeEnv,
) -> Result<()> {
    ctx.output_logits.clear();
    ctx.output_embeddings.clear();
    ctx.sampled_tokens.clear();

    let model = Arc::clone(&ctx.model);
    let prefix = env.parse_correction_prefix(model.block_size)?;
    let trace_on = env.get_bool("DSPARK_TEST_TRACE");
    let two_rows = env.get_bool("LLAMA_DSPARK_SNR_TWO_ROWS");
    let balanced = env.get_bool("LLAMA_DSPARK_BALANCED_OUTPUT");
    let reuse_mask = env.get_bool("LLAMA_DSPARK_REUSE_MASK");

    let staged_rows = ctx.staged.as_ref().map_or(0, |s| s.positions.len());
    let draft_rows: Vec<&BatchRow> = batch.rows.iter().skip(staged_rows).collect();
    if reuse_mask
        && draft_rows.len() > 1
        && draft_rows[0].token != model.mask_token_id
        && draft_rows[1].token != model.mask_token_id
    {
        return Err(invalid(
            "runtime-error:anchor followed by MASK expected in reuse-mask mode",
        ));
    }

    let n_embd = model.n_embd;
    let mut seq_hidden = Vec::with_capacity(batch.rows.len());

    if let Some(staged) = &ctx.staged {
        let fc_w = model.tensor_f32("dspark.fc.weight")?;
        let mut fc_flat = Vec::with_capacity(staged.positions.len() * n_embd);
        for chunk in staged.features.chunks_exact(staged.width) {
            let proj = matvec_f64_round(&fc_w, chunk)?;
            fc_flat.extend_from_slice(&proj);
            seq_hidden.push(proj);
        }
        record_trace(ctx, traces, trace_on, "dspark_fc", &fc_flat);
    }

    let snr_w1 = tensor_slice(&model, "dspark.log_snr_fc1.weight")?;
    let snr_w2 = tensor_slice(&model, "dspark.log_snr_fc2.weight")?;
    let mut fc2_all = Vec::new();
    let mut draft_snr_all = Vec::new();
    let mut two_row_cache: BTreeMap<bool, Vec<f32>> = BTreeMap::new();

    for drow in &draft_rows {
        let mut emb = embed_token(&model, drow.token)?;
        let is_mask = drow.token == model.mask_token_id;
        if let (Some(w1), Some(w2)) = (&snr_w1, &snr_w2) {
            let cond: &[f32] = if let Some(cached) = two_row_cache.get(&is_mask) {
                cached
            } else {
                let snr_val = if is_mask { 0.0_f32 } else { 1.0_f32 };
                let in_dim = w1.len() / n_embd.max(1);
                let snr_in = vec![snr_val; in_dim.max(1)];
                let h1 = matvec_f64_round(w1, &snr_in)?;
                let silu: Vec<f32> = h1.into_iter().map(|x| x / (1.0 + (-x).exp())).collect();
                let h2 = matvec_f64_round(w2, &silu)?;
                two_row_cache.entry(is_mask).or_insert(h2)
            };
            fc2_all.extend_from_slice(cond);
            for (ei, &ci) in emb.iter_mut().zip(cond) {
                *ei += ci;
            }
        }
        draft_snr_all.extend_from_slice(&emb);
        seq_hidden.push(emb);
    }

    if two_rows {
        let mut reduced = Vec::new();
        for flag in [false, true] {
            if let Some(v) = two_row_cache.get(&flag) {
                reduced.extend_from_slice(v);
            }
        }
        record_trace(ctx, traces, trace_on, "dspark_log_snr_fc2", &reduced);
        record_trace(ctx, traces, trace_on, "dspark_log_snr_expanded", &fc2_all);
    } else {
        record_trace(ctx, traces, trace_on, "dspark_log_snr_fc2", &fc2_all);
    }
    record_trace(
        ctx,
        traces,
        trace_on,
        "dspark_draft_embd_snr",
        &draft_snr_all,
    );

    let norm_all: Vec<f32> = seq_hidden.iter().flatten().copied().collect();
    record_trace(ctx, traces, trace_on, "dspark_hidden_norm", &norm_all);

    let after_trunk = run_dspark_trunk(&model, ctx, &batch.rows, seq_hidden)?;
    let draft_slice = after_trunk.get(staged_rows..).unwrap_or(&[]);
    let out_norm_w = tensor_slice(&model, "output_norm.weight")?;
    let mut result_norm = Vec::with_capacity(draft_slice.len());
    for h in draft_slice {
        result_norm.push(rms_norm(h, out_norm_w.as_deref(), model.rms_eps)?);
    }
    let flat_result_norm: Vec<f32> = result_norm.iter().flatten().copied().collect();
    record_trace(ctx, traces, trace_on, "result_norm", &flat_result_norm);

    let req_rows = draft_rows.iter().filter(|r| r.logits).count();
    if let Some(keep) = prefix
        && req_rows > keep
    {
        return Err(invalid(
            "runtime-error:correction prefix smaller than requested output rows",
        ));
    }

    let n_keep = if model.graph_corrected {
        req_rows.max(1).min(draft_rows.len())
    } else {
        draft_rows.len()
    };

    if model.graph_corrected && !draft_rows.is_empty() {
        run_dspark_correction_head(
            ctx,
            traces,
            trace_on,
            balanced,
            &draft_rows[..n_keep],
            &result_norm[..n_keep],
        )?;
    } else {
        for (i, drow) in draft_rows.iter().enumerate() {
            if drow.logits {
                let mut logits = project_logits(&model, &result_norm[i], &ctx.lora_adapters)?;
                let mask_idx = usize::try_from(model.mask_token_id.max(0)).unwrap_or(0);
                if mask_idx < logits.len() {
                    logits[mask_idx] = -1e6;
                }
                let best_id = argmax_f32(&logits)?;
                ctx.sampled_tokens.push(best_id);
                ctx.output_logits.push(logits);
            }
        }
    }
    Ok(())
}

fn dspark_layer_attn(
    model: &ModelData,
    ctx: &ContextState,
    rows: &[BatchRow],
    qs: &[Vec<f32>],
    ks: &[Vec<f32>],
    vs: &[Vec<f32>],
    row_idx: usize,
) -> Result<Vec<f32>> {
    let head_dim = (model.n_embd / model.n_head.max(1)).max(2);
    let scale = (f64::from(u32::try_from(head_dim)?)).sqrt().recip();
    let mut attn = vec![0.0_f32; model.n_embd];
    for h_idx in 0..model.n_head.max(1) {
        let q_h = &qs[row_idx][h_idx * head_dim..(h_idx + 1) * head_dim];
        let mut scores = Vec::with_capacity(rows.len());
        let mut max_s = f64::NEG_INFINITY;
        for (j, k_all) in ks.iter().enumerate() {
            if ctx.causal_attention && rows[j].pos > rows[row_idx].pos {
                scores.push(f64::NEG_INFINITY);
                continue;
            }
            let k_h = &k_all[h_idx * head_dim..(h_idx + 1) * head_dim];
            let s = q_h
                .iter()
                .zip(k_h)
                .map(|(&a, &b)| f64::from(a) * f64::from(b))
                .sum::<f64>()
                * scale;
            if s > max_s {
                max_s = s;
            }
            scores.push(s);
        }
        let exps: Vec<f64> = scores.into_iter().map(|s| (s - max_s).exp()).collect();
        let denom: f64 = exps.iter().sum();
        let mut acc = vec![0.0_f64; head_dim];
        for (w, v_all) in exps.into_iter().zip(vs) {
            let p = w / denom;
            let v_h = &v_all[h_idx * head_dim..(h_idx + 1) * head_dim];
            for (a, &vi) in acc.iter_mut().zip(v_h) {
                *a = p.mul_add(f64::from(vi), *a);
            }
        }
        for (dst, a) in attn[h_idx * head_dim..(h_idx + 1) * head_dim]
            .iter_mut()
            .zip(acc)
        {
            *dst = f64_to_f32(a);
        }
    }
    Ok(attn)
}

fn run_dspark_trunk(
    model: &Arc<ModelData>,
    ctx: &mut ContextState,
    rows: &[BatchRow],
    mut states: Vec<Vec<f32>>,
) -> Result<Vec<Vec<f32>>> {
    let head_dim = (model.n_embd / model.n_head.max(1)).max(2);
    let seq_id = rows
        .first()
        .and_then(|r| r.seq_ids.first())
        .copied()
        .unwrap_or(0);
    for layer in 0..model.n_layer.min(32) {
        let anorm_w = tensor_slice(model, &format!("blk.{layer}.attn_norm.weight"))?;
        let mut qs = Vec::with_capacity(rows.len());
        let mut ks = Vec::with_capacity(rows.len());
        let mut vs = Vec::with_capacity(rows.len());
        for (idx, h) in states.iter().enumerate() {
            let x = rms_norm(h, anorm_w.as_deref(), model.rms_eps)?;
            let mut q = linear_with_lora(
                model,
                &format!("blk.{layer}.attn_q.weight"),
                &x,
                model.n_embd,
                &[],
            )?;
            let mut k = linear_with_lora(
                model,
                &format!("blk.{layer}.attn_k.weight"),
                &x,
                model.n_embd,
                &[],
            )?;
            let v = linear_with_lora(
                model,
                &format!("blk.{layer}.attn_v.weight"),
                &x,
                model.n_embd,
                &[],
            )?;
            apply_rope(&mut q, rows[idx].pos, head_dim, model.rope_freq_base);
            apply_rope(&mut k, rows[idx].pos, head_dim, model.rope_freq_base);
            qs.push(q);
            ks.push(k);
            vs.push(v);
        }
        let mut next_states = Vec::with_capacity(rows.len());
        for (i, h) in states.iter().enumerate() {
            let attn = dspark_layer_attn(model, ctx, rows, &qs, &ks, &vs, i)?;
            let o_proj = linear_with_lora(
                model,
                &format!("blk.{layer}.attn_output.weight"),
                &attn,
                model.n_embd,
                &[],
            )?;
            let mut h_mid: Vec<f32> = h.iter().zip(o_proj).map(|(&a, b)| a + b).collect();
            let fnorm_w = tensor_slice(model, &format!("blk.{layer}.ffn_norm.weight"))?;
            let f_in = rms_norm(&h_mid, fnorm_w.as_deref(), model.rms_eps)?;
            let g = linear_with_lora(
                model,
                &format!("blk.{layer}.ffn_gate.weight"),
                &f_in,
                model.n_ff,
                &[],
            )?;
            let u = linear_with_lora(
                model,
                &format!("blk.{layer}.ffn_up.weight"),
                &f_in,
                model.n_ff,
                &[],
            )?;
            let gated: Vec<f32> = g
                .into_iter()
                .zip(u)
                .map(|(gv, uv)| (gv / (1.0 + (-gv).exp())) * uv)
                .collect();
            let d = linear_with_lora(
                model,
                &format!("blk.{layer}.ffn_down.weight"),
                &gated,
                model.n_embd,
                &[],
            )?;
            for (hm, dv) in h_mid.iter_mut().zip(d) {
                *hm += dv;
            }
            next_states.push(h_mid);
        }
        states = next_states;
    }
    let _ = ctx.sequences.entry(seq_id).or_default();
    Ok(states)
}

fn run_dspark_correction_head(
    ctx: &mut ContextState,
    traces: &mut BTreeMap<u64, TraceCollector>,
    trace_on: bool,
    balanced: bool,
    draft_rows: &[&BatchRow],
    result_norm: &[Vec<f32>],
) -> Result<()> {
    let model = Arc::clone(&ctx.model);
    let emb_all = model.tensor_f32("token_embd.weight")?;
    let hnorm = model.tensor_f32("dspark.correction_hidden_norm.weight")?;
    let enorm = model.tensor_f32("dspark.correction_embed_norm.weight")?;
    let c_gate = model.tensor_f32("dspark.correction_gate.weight")?;
    let c_up = model.tensor_f32("dspark.correction_up.weight")?;
    let c_down = model.tensor_f32("dspark.correction_down.weight")?;
    let m_a = model.tensor_f32("dspark.markov_head_a.weight")?;
    let m_b = model.tensor_f32("dspark.markov_head_b.weight")?;
    let head = if model.output_buffer.is_empty() {
        &emb_all
    } else {
        model.output_buffer.as_slice()
    };
    let embd = model.n_embd;
    let rank = m_a.len() / model.vocab_size.max(1);
    let mask_idx = usize::try_from(model.mask_token_id.max(0)).unwrap_or(0);
    let mut previous = usize::try_from(draft_rows[0].token)?;
    let mut bal_trace = Vec::new();

    for (slot, (drow, rnorm)) in draft_rows.iter().zip(result_norm).enumerate() {
        let mut hidden = rnorm.clone();
        let mut input = rms_norm(&hidden, Some(&hnorm), 1e-6)?;
        let prev_emb = emb_all
            .get(previous * embd..(previous + 1) * embd)
            .ok_or_else(|| invalid("previous embedding out of bounds"))?;
        input.extend(rms_norm(prev_emb, Some(&enorm), 1e-6)?);
        let mut gate = matvec_f64_round(&c_gate, &input)?;
        let up = matvec_f64_round(&c_up, &input)?;
        for (gv, uv) in gate.iter_mut().zip(up) {
            *gv = *gv / (1.0 + (-*gv).exp()) * uv;
        }
        let delta = matvec_f64_round(&c_down, &gate)?;
        for (hv, dv) in hidden.iter_mut().zip(delta) {
            *hv += dv;
        }
        if balanced {
            bal_trace.extend_from_slice(&hidden);
        }
        let mut logits = matvec_f64_round(head, &hidden)?;
        let tok_u = usize::try_from(drow.token)?;
        let a_slice = m_a
            .get(tok_u * rank..(tok_u + 1) * rank)
            .ok_or_else(|| invalid("markov index out of bounds"))?;
        let bias = matvec_f64_round(&m_b, a_slice)?;
        for (lv, bv) in logits.iter_mut().zip(bias) {
            *lv += bv;
        }
        record_trace(
            ctx,
            traces,
            trace_on,
            &format!("dspark_corr_unmasked-{slot}"),
            &logits,
        );
        if mask_idx < logits.len() {
            logits[mask_idx] = -1e6;
        }
        record_trace(
            ctx,
            traces,
            trace_on,
            &format!("dspark_corr_masked-{slot}"),
            &logits,
        );
        let best = argmax_f32(&logits)?;
        previous = usize::try_from(best)?;
        ctx.sampled_tokens.push(best);
        if drow.logits {
            ctx.output_logits.push(logits);
        }
    }
    if balanced {
        record_trace(ctx, traces, trace_on, "dspark_balanced_output", &bal_trace);
    }
    Ok(())
}

fn record_trace(
    ctx: &ContextState,
    traces: &mut BTreeMap<u64, TraceCollector>,
    trace_on: bool,
    name: &str,
    data: &[f32],
) {
    if !trace_on {
        return;
    }
    if let Some(tid) = ctx.trace_id
        && let Some(collector) = traces.get_mut(&tid)
    {
        collector.record(name, data);
    }
}

pub(super) fn argmax_f32(row: &[f32]) -> Result<i32> {
    if row.is_empty() {
        return Err(invalid("empty logits slice in argmax"));
    }
    let mut best = 0_usize;
    for i in 1..row.len() {
        if row[i] > row[best] {
            best = i;
        }
    }
    Ok(i32::try_from(best)?)
}

const GGML_OP_NAMES: [&str; 93] = [
    "NONE",
    "DUP",
    "ADD",
    "ADD_ID",
    "ADD1",
    "ACC",
    "SUB",
    "MUL",
    "DIV",
    "SQR",
    "SQRT",
    "LOG",
    "SIN",
    "COS",
    "SUM",
    "SUM_ROWS",
    "MEAN",
    "ARGMAX",
    "COUNT_EQUAL",
    "REPEAT",
    "REPEAT_BACK",
    "CONCAT",
    "SILU_BACK",
    "NORM",
    "RMS_NORM",
    "RMS_NORM_BACK",
    "GROUP_NORM",
    "L2_NORM",
    "MUL_MAT",
    "MUL_MAT_ID",
    "OUT_PROD",
    "SCALE",
    "SET",
    "CPY",
    "CONT",
    "RESHAPE",
    "VIEW",
    "PERMUTE",
    "TRANSPOSE",
    "GET_ROWS",
    "GET_ROWS_BACK",
    "SET_ROWS",
    "DIAG",
    "DIAG_MASK_INF",
    "DIAG_MASK_ZERO",
    "SOFT_MAX",
    "SOFT_MAX_BACK",
    "ROPE",
    "ROPE_BACK",
    "CLAMP",
    "CONV_TRANSPOSE_1D",
    "IM2COL",
    "IM2COL_BACK",
    "CONV_2D",
    "CONV_3D",
    "CONV_2D_DW",
    "CONV_TRANSPOSE_2D",
    "POOL_1D",
    "POOL_2D",
    "POOL_2D_BACK",
    "UPSCALE",
    "PAD",
    "PAD_REFLECT_1D",
    "ROLL",
    "ARANGE",
    "TIMESTEP_EMBEDDING",
    "ARGSORT",
    "TOP_K",
    "LEAKY_RELU",
    "FILL",
    "FLASH_ATTN_EXT",
    "FLASH_ATTN_BACK",
    "SSM_CONV",
    "SSM_SCAN",
    "WIN_PART",
    "WIN_UNPART",
    "GET_REL_POS",
    "ADD_REL_POS",
    "RWKV_WKV6",
    "GATED_LINEAR_ATTN",
    "RWKV_WKV7",
    "SOLVE_TRI",
    "UNARY",
    "MAP_CUSTOM1",
    "MAP_CUSTOM2",
    "MAP_CUSTOM3",
    "CUSTOM",
    "CROSS_ENTROPY_LOSS",
    "CROSS_ENTROPY_LOSS_BACK",
    "OPT_STEP_ADAMW",
    "OPT_STEP_SGD",
    "GLU",
    "COUNT",
];

pub(super) fn context_limits(ctx: &ContextState) -> Value {
    let mut ops = serde_json::Map::new();
    for (idx, name) in GGML_OP_NAMES.into_iter().enumerate() {
        ops.insert(name.to_owned(), json!(idx));
    }
    json!({
        "Ctx": ctx.n_ctx,
        "Ubatch": ctx.n_ubatch,
        "SeqMax": ctx.n_seq_max,
        "Ops": Value::Object(ops),
    })
}

pub(super) fn context_graph_reserve(ctx: &ContextState, tokens: u64) -> Value {
    let embd = i64::try_from(ctx.model.n_embd).unwrap_or(64);
    let n_tok = i64::try_from(tokens.max(1)).unwrap_or(1);
    let nodes = vec![
        json!({
            "Index": 0, "Op": 39, "Type": 0, "NE": [embd, n_tok, 1, 1], "NB": [4, embd * 4, embd * n_tok * 4, embd * n_tok * 4],
            "OpParamsHex": "00000000", "Name": "inp_embd", "Sources": [Value::Null]
        }),
        json!({
            "Index": 1, "Op": 24, "Type": 0, "NE": [embd, n_tok, 1, 1], "NB": [4, embd * 4, embd * n_tok * 4, embd * n_tok * 4],
            "OpParamsHex": "00000000", "Name": "attn_norm-0", "Sources": [0]
        }),
        json!({
            "Index": 2, "Op": 28, "Type": 0, "NE": [embd, n_tok, 1, 1], "NB": [4, embd * 4, embd * n_tok * 4, embd * n_tok * 4],
            "OpParamsHex": "00000000", "Name": "Qcur-0", "Sources": [1]
        }),
        json!({
            "Index": 3, "Op": 28, "Type": 0, "NE": [embd, n_tok, 1, 1], "NB": [4, embd * 4, embd * n_tok * 4, embd * n_tok * 4],
            "OpParamsHex": "00000000", "Name": "Kcur-0", "Sources": [1]
        }),
        json!({
            "Index": 4, "Op": 28, "Type": 0, "NE": [embd, n_tok, 1, 1], "NB": [4, embd * 4, embd * n_tok * 4, embd * n_tok * 4],
            "OpParamsHex": "00000000", "Name": "Vcur-0", "Sources": [1]
        }),
        json!({
            "Index": 5, "Op": 47, "Type": 0, "NE": [embd, n_tok, 1, 1], "NB": [4, embd * 4, embd * n_tok * 4, embd * n_tok * 4],
            "OpParamsHex": "00000000", "Name": "Q_rope-0", "Sources": [2]
        }),
        json!({
            "Index": 6, "Op": 45, "Type": 0, "NE": [n_tok, n_tok, 1, 1], "NB": [4, n_tok * 4, n_tok * n_tok * 4, n_tok * n_tok * 4],
            "OpParamsHex": "00000000", "Name": "kq_soft_max-0", "Sources": [5, 3]
        }),
        json!({
            "Index": 7, "Op": 2, "Type": 0, "NE": [embd, n_tok, 1, 1], "NB": [4, embd * 4, embd * n_tok * 4, embd * n_tok * 4],
            "OpParamsHex": "00000000", "Name": "result_norm", "Sources": [0, 6]
        }),
    ];
    json!({
        "Reserved": true,
        "Nodes": nodes,
    })
}
