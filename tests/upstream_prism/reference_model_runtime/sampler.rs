//! Clean-room sampler chains, `MT19937` distribution sampling, observers, and `bs-*` ops.
//!
//! No `bonsai_prism_runtime`, `bonsai_gguf`, or native GGML libraries are linked or called.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use serde_json::{Value, json};

use super::gguf_model::{ModelData, checked_session_file, f64_to_f32};
use crate::{Result, invalid};

const MAX_CANDIDATES: usize = 1_048_576;
const MAX_BACKEND_CANDIDATES: usize = 131_072;
const MAX_HANDLES: usize = 128;

pub(super) type SamplerEventCb<'a> = &'a mut dyn FnMut(&Value) -> Result<Value>;

#[derive(Clone)]
pub(super) struct Mt19937 {
    mt: [u32; 624],
    index: usize,
}

impl Mt19937 {
    pub(super) fn new(seed: u32) -> Self {
        let mut mt = [0_u32; 624];
        mt[0] = seed;
        for i in 1..624 {
            let prev = mt[i - 1];
            mt[i] = 1_812_433_253_u32
                .wrapping_mul(prev ^ (prev >> 30))
                .wrapping_add(u32::try_from(i).unwrap_or(0));
        }
        Self { mt, index: 624 }
    }

    pub(super) fn next_u32(&mut self) -> u32 {
        if self.index >= 624 {
            self.twist();
        }
        let mut y = self.mt[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^ (y >> 18)
    }

    fn twist(&mut self) {
        for i in 0..624 {
            let y = (self.mt[i] & 0x8000_0000) | (self.mt[(i + 1) % 624] & 0x7fff_ffff);
            let next = self.mt[(i + 397) % 624] ^ (y >> 1);
            self.mt[i] = if y & 1 != 0 { next ^ 0x9908_b0df } else { next };
        }
        self.index = 0;
    }

    pub(super) fn next_f64_unit(&mut self) -> f64 {
        let a = f64::from(self.next_u32() >> 5);
        let b = f64::from(self.next_u32() >> 6);
        a.mul_add(67_108_864.0, b) / 9_007_199_254_740_992.0
    }
}

#[derive(Clone, Copy)]
pub(super) struct TokenCandidate {
    pub(super) id: i32,
    pub(super) logit: f32,
    pub(super) p: f32,
}

#[derive(Clone)]
pub(super) struct CandidateBuffer {
    pub(super) backing: Vec<TokenCandidate>,
    pub(super) offset: usize,
    pub(super) size: usize,
    pub(super) selected: i64,
    pub(super) sorted: bool,
}

impl CandidateBuffer {
    pub(super) fn from_logits(logits: &[f32]) -> Result<Self> {
        let mut backing = Vec::with_capacity(logits.len());
        for (idx, &logit) in logits.iter().enumerate() {
            backing.push(TokenCandidate {
                id: i32::try_from(idx)?,
                logit,
                p: 0.0,
            });
        }
        let size = backing.len();
        Ok(Self {
            backing,
            offset: 0,
            size,
            selected: -1,
            sorted: false,
        })
    }

    pub(super) fn active_slice(&self) -> &[TokenCandidate] {
        &self.backing[self.offset..self.offset + self.size]
    }

    pub(super) fn active_mut(&mut self) -> &mut [TokenCandidate] {
        let end = self.offset + self.size;
        &mut self.backing[self.offset..end]
    }

    fn sort_active_desc(&mut self) {
        if !self.sorted {
            self.active_mut()
                .sort_by(|a, b| b.logit.total_cmp(&a.logit));
            self.sorted = true;
        }
    }

    fn apply_softmax(&mut self, with_probs: bool) {
        self.sort_active_desc();
        let active = self.active_mut();
        if active.is_empty() {
            return;
        }
        let max_l = f64::from(active[0].logit);
        let mut sum = 0.0_f64;
        for item in active.iter_mut() {
            let e = (f64::from(item.logit) - max_l).exp();
            if with_probs {
                item.p = f64_to_f32(e);
            }
            sum += e;
        }
        if with_probs && sum > 0.0 {
            for item in active.iter_mut() {
                item.p = f64_to_f32(f64::from(item.p) / sum);
            }
        }
    }

    pub(super) fn selected_token(&self) -> Result<i32> {
        let active = self.active_slice();
        if active.is_empty() {
            return Err(invalid("cannot select token from empty candidates"));
        }
        if self.selected >= 0 {
            let idx = usize::try_from(self.selected)?;
            if idx < active.len() {
                return Ok(active[idx].id);
            }
        }
        Ok(active[0].id)
    }

    fn snapshot_json(&self) -> Value {
        json!({
            "BufferSize": self.backing.len(),
            "Size": self.size,
            "Offset": self.offset,
            "Selected": self.selected,
            "Sorted": self.sorted,
        })
    }
}

#[derive(Clone)]
pub(super) enum SamplerNode {
    Chain(Vec<Self>),
    Greedy,
    Dist(Mt19937),
    Temp(f32),
    TempExt {
        temp: f32,
        delta: f32,
        exponent: f32,
    },
    TopK(i32),
    TopP {
        p: f32,
        min_keep: usize,
    },
    MinP {
        p: f32,
        min_keep: usize,
    },
    Typical {
        p: f32,
        min_keep: usize,
    },
    Xtc {
        p: f32,
        threshold: f32,
        min_keep: usize,
        rng: Mt19937,
    },
    LogitBias(Vec<(i32, f32)>),
    Penalties {
        last_n: i32,
        repeat: f32,
        freq: f32,
        presence: f32,
        history: Vec<i32>,
    },
    Dry {
        mult: f32,
        base: f32,
        allowed_len: i32,
        last_n: i32,
        breakers: Vec<Vec<i32>>,
        history: Vec<i32>,
    },
    TopNSigma(f32),
    Observer {
        label: u32,
    },
}

impl SamplerNode {
    pub(super) fn has_observer(&self) -> bool {
        match self {
            Self::Observer { .. } => true,
            Self::Chain(items) => items.iter().any(Self::has_observer),
            _ => false,
        }
    }

    pub(super) fn notify_backend_init(
        &self,
        outputs: u32,
        cb: &mut dyn FnMut(&Value) -> Result<Value>,
    ) -> Result<bool> {
        match self {
            Self::Observer { label } => {
                let res = cb(&json!({
                    "Kind": "backend-init",
                    "Sampler": label,
                    "OutputsMaxPerSequence": outputs,
                    "BufferTypeName": "CPU",
                }))?;
                Ok(res["Accept"].as_bool().unwrap_or(false))
            }
            Self::Chain(items) => {
                let mut all = !items.is_empty();
                for item in items {
                    all &= item.notify_backend_init(outputs, cb)?;
                }
                Ok(all)
            }
            _ => Ok(true),
        }
    }

    pub(super) fn accept(&mut self, token: i32) {
        match self {
            Self::Chain(items) => {
                for item in items {
                    item.accept(token);
                }
            }
            Self::Penalties {
                history, last_n, ..
            } if *last_n != 0 => {
                history.push(token);
                let cap = usize::try_from(*last_n).unwrap_or(2048);
                if *last_n > 0 && history.len() > cap {
                    history.remove(0);
                }
            }
            Self::Dry {
                history, last_n, ..
            } if *last_n != 0 => {
                history.push(token);
                let cap = usize::try_from(*last_n).unwrap_or(2048);
                if *last_n > 0 && history.len() > cap {
                    history.remove(0);
                }
            }
            _ => {}
        }
    }

    pub(super) fn apply(
        &mut self,
        buf: &mut CandidateBuffer,
        cb: &mut Option<SamplerEventCb<'_>>,
    ) -> Result<()> {
        match self {
            Self::Chain(items) => {
                for item in items {
                    item.apply(buf, cb)?;
                }
                Ok(())
            }
            Self::Greedy => {
                let mut best_i = 0_usize;
                let active = buf.active_slice();
                for (i, c) in active.iter().enumerate().skip(1) {
                    if c.logit > active[best_i].logit {
                        best_i = i;
                    }
                }
                buf.selected = i64::try_from(best_i)?;
                Ok(())
            }
            Self::Dist(rng) => apply_dist(rng, buf),
            Self::Temp(t) => {
                apply_temp(*t, buf);
                Ok(())
            }
            Self::TempExt {
                temp,
                delta,
                exponent,
            } => apply_temp_ext(*temp, *delta, *exponent, buf),
            Self::TopK(k) => apply_top_k(*k, buf),
            Self::TopP { p, min_keep } => {
                apply_top_p(*p, *min_keep, buf);
                Ok(())
            }
            Self::MinP { p, min_keep } => {
                apply_min_p(*p, *min_keep, buf);
                Ok(())
            }
            Self::Typical { p, min_keep } => {
                apply_typical(*p, *min_keep, buf);
                Ok(())
            }
            Self::Xtc {
                p,
                threshold,
                min_keep,
                rng,
            } => {
                apply_xtc(*p, *threshold, *min_keep, rng, buf);
                Ok(())
            }
            Self::LogitBias(biases) => {
                apply_logit_bias(biases, buf);
                Ok(())
            }
            Self::Penalties {
                repeat,
                freq,
                presence,
                history,
                ..
            } => {
                apply_penalties(*repeat, *freq, *presence, history, buf);
                Ok(())
            }
            Self::Dry {
                mult,
                base,
                allowed_len,
                history,
                breakers,
                ..
            } => {
                apply_dry(*mult, *base, *allowed_len, history, breakers, buf);
                Ok(())
            }
            Self::TopNSigma(n) => {
                apply_top_n_sigma(*n, buf);
                Ok(())
            }
            Self::Observer { label } => {
                if let Some(sink) = cb.as_deref_mut() {
                    let _ = sink(&json!({ "Kind": "apply", "Sampler": label }))?;
                }
                Ok(())
            }
        }
    }

    pub(super) fn sample_logits(
        &mut self,
        logits: &[f32],
        cb: &mut Option<SamplerEventCb<'_>>,
    ) -> Result<(i32, CandidateBuffer)> {
        let mut buf = CandidateBuffer::from_logits(logits)?;
        self.apply(&mut buf, cb)?;
        let tok = buf.selected_token()?;
        Ok((tok, buf))
    }
}

fn apply_dist(rng: &mut Mt19937, buf: &mut CandidateBuffer) -> Result<()> {
    buf.apply_softmax(true);
    let active = buf.active_slice();
    if active.is_empty() {
        return Err(invalid("dist sampler empty candidates"));
    }
    let u = rng.next_f64_unit();
    let mut cdf = 0.0_f64;
    let mut chosen = active.len() - 1;
    for (idx, item) in active.iter().enumerate() {
        cdf += f64::from(item.p);
        if cdf > u {
            chosen = idx;
            break;
        }
    }
    buf.selected = i64::try_from(chosen)?;
    Ok(())
}

fn apply_temp(temp: f32, buf: &mut CandidateBuffer) {
    if temp <= 0.0 {
        let active = buf.active_slice();
        let mut best_i = 0_usize;
        for (i, c) in active.iter().enumerate().skip(1) {
            if c.logit > active[best_i].logit {
                best_i = i;
            }
        }
        buf.backing[buf.offset] = active[best_i];
        buf.size = 1;
        buf.selected = 0;
        buf.sorted = true;
        return;
    }
    for c in buf.active_mut() {
        c.logit /= temp;
    }
}

fn apply_temp_ext(temp: f32, delta: f32, exponent: f32, buf: &mut CandidateBuffer) -> Result<()> {
    if temp <= 0.0 || delta <= 0.0 {
        apply_temp(temp, buf);
        return Ok(());
    }
    buf.apply_softmax(true);
    let min_temp = (temp - delta).max(0.0);
    let max_temp = temp + delta;
    let mut entropy = 0.0_f64;
    for c in buf.active_slice() {
        if c.p > 0.0 {
            entropy = f64::mul_add(f64::from(c.p), -f64::from(c.p).ln(), entropy);
        }
    }
    let count = buf.size.max(2);
    let max_ent = f64::from(u32::try_from(count)?).ln();
    let norm = if max_ent > 0.0 {
        (entropy / max_ent).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let eff = f64::mul_add(
        f64::from(max_temp - min_temp),
        norm.powf(f64::from(exponent)),
        f64::from(min_temp),
    );
    let eff32 = f64_to_f32(eff.max(1e-6));
    for c in buf.active_mut() {
        c.logit /= eff32;
    }
    buf.apply_softmax(true);
    Ok(())
}

fn apply_top_k(k: i32, buf: &mut CandidateBuffer) -> Result<()> {
    if k <= 0 {
        return Ok(());
    }
    buf.sort_active_desc();
    let keep = usize::try_from(k)?.min(buf.size).max(1);
    buf.size = keep;
    Ok(())
}

fn apply_top_p(p: f32, min_keep: usize, buf: &mut CandidateBuffer) {
    if p >= 1.0 {
        return;
    }
    buf.apply_softmax(true);
    let mut cum = 0.0_f32;
    let mut last_idx = buf.size;
    for (i, c) in buf.active_slice().iter().enumerate() {
        cum += c.p;
        if cum >= p && i + 1 >= min_keep.max(1) {
            last_idx = i + 1;
            break;
        }
    }
    buf.size = last_idx.max(min_keep.max(1)).min(buf.size);
}

fn apply_min_p(p: f32, min_keep: usize, buf: &mut CandidateBuffer) {
    if p <= 0.0 || buf.size <= 1 {
        return;
    }
    buf.apply_softmax(true);
    let max_logit = buf.active_slice()[0].logit;
    let thresh = max_logit + p.ln();
    let mut keep = 0_usize;
    for (i, c) in buf.active_slice().iter().enumerate() {
        if c.logit >= thresh || i < min_keep {
            keep = i + 1;
        } else {
            break;
        }
    }
    buf.size = keep.max(min_keep.max(1)).min(buf.size);
}

fn apply_typical(p: f32, min_keep: usize, buf: &mut CandidateBuffer) {
    if p >= 1.0 || buf.size <= 1 {
        return;
    }
    buf.apply_softmax(true);
    let active = buf.active_slice().to_vec();
    let mut entropy = 0.0_f64;
    for c in &active {
        if c.p > 0.0 {
            entropy = f64::mul_add(f64::from(c.p), -f64::from(c.p).ln(), entropy);
        }
    }
    let mut scored: Vec<(f64, TokenCandidate)> = active
        .into_iter()
        .map(|c| {
            let neg_ln = if c.p > 0.0 {
                -f64::from(c.p).ln()
            } else {
                f64::INFINITY
            };
            ((neg_ln - entropy).abs(), c)
        })
        .collect();
    scored.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut sum = 0.0_f32;
    let mut out = Vec::new();
    let min_out = min_keep.max(1);
    for (_, c) in scored {
        out.push(c);
        sum += c.p;
        if sum > p && out.len() >= min_out {
            break;
        }
    }
    buf.backing[buf.offset..buf.offset + out.len()].copy_from_slice(&out);
    buf.size = out.len();
    buf.sorted = false;
}

fn apply_xtc(
    p: f32,
    threshold: f32,
    min_keep: usize,
    rng: &mut Mt19937,
    buf: &mut CandidateBuffer,
) {
    if p <= 0.0 || threshold > 0.5 || buf.size <= min_keep {
        return;
    }
    if f64_to_f32(rng.next_f64_unit()) >= p {
        return;
    }
    buf.apply_softmax(true);
    let active = buf.active_slice();
    let drop_n = active
        .iter()
        .take_while(|c| c.p >= threshold)
        .count()
        .min(buf.size.saturating_sub(min_keep));
    if drop_n > 0 && active[drop_n - 1].p >= threshold {
        buf.offset += drop_n;
        buf.size -= drop_n;
    }
}

fn apply_logit_bias(biases: &[(i32, f32)], buf: &mut CandidateBuffer) {
    let map: BTreeMap<i32, f32> = biases.iter().copied().collect();
    for c in buf.active_mut() {
        if let Some(&bias) = map.get(&c.id) {
            c.logit += bias;
        }
    }
    buf.sorted = false;
}

fn apply_penalties(
    repeat: f32,
    freq: f32,
    presence: f32,
    history: &[i32],
    buf: &mut CandidateBuffer,
) {
    if history.is_empty() {
        return;
    }
    let mut counts: BTreeMap<i32, usize> = BTreeMap::new();
    for &t in history {
        *counts.entry(t).or_insert(0) += 1;
    }
    for c in buf.active_mut() {
        if let Some(&count) = counts.get(&c.id) {
            if c.logit <= 0.0 {
                c.logit *= repeat;
            } else {
                c.logit /= repeat;
            }
            let cnt = f64_to_f32(f64::from(u32::try_from(count).unwrap_or(0)));
            c.logit -= cnt.mul_add(freq, presence);
        }
    }
    buf.sorted = false;
}

fn apply_dry(
    mult: f32,
    base: f32,
    allowed_len: i32,
    history: &[i32],
    breakers: &[Vec<i32>],
    buf: &mut CandidateBuffer,
) {
    if mult == 0.0 || history.is_empty() {
        return;
    }
    let allowed = usize::try_from(allowed_len.max(0)).unwrap_or(0);
    let tail_breaker = breakers
        .iter()
        .any(|b| !b.is_empty() && history.ends_with(b));
    if tail_breaker {
        return;
    }
    let last = history[history.len() - 1];
    for c in buf.active_mut() {
        if history.contains(&c.id) {
            let mut max_match = 1_usize;
            for &tok in history.iter().take(history.len().saturating_sub(1)) {
                if tok == last {
                    max_match = max_match.max(2);
                }
            }
            if max_match > allowed {
                let exp = i32::try_from(max_match - allowed).unwrap_or(0);
                c.logit -= f64_to_f32(f64::from(mult) * f64::from(base).powi(exp));
            }
        }
    }
    buf.sorted = false;
}

fn apply_top_n_sigma(n: f32, buf: &mut CandidateBuffer) {
    if n <= 0.0 || buf.size <= 1 {
        return;
    }
    buf.sort_active_desc();
    let finite: Vec<f64> = buf
        .active_slice()
        .iter()
        .map(|c| f64::from(c.logit))
        .filter(|v| v.is_finite())
        .collect();
    if finite.is_empty() {
        return;
    }
    let len = f64::from(u32::try_from(finite.len()).unwrap_or(1));
    let mean = finite.iter().sum::<f64>() / len;
    let var = finite.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / len;
    let thresh = f64::mul_add(f64::from(n), -var.sqrt(), finite[0]);
    let keep = buf
        .active_slice()
        .iter()
        .take_while(|c| f64::from(c.logit) >= thresh)
        .count()
        .max(1);
    buf.size = keep;
}

pub(super) struct BackendSamplerRegistry {
    next: u64,
    pub(super) samplers: BTreeMap<u64, SamplerNode>,
    candidates: BTreeMap<u64, CandidateBuffer>,
}

impl BackendSamplerRegistry {
    pub(super) const fn new() -> Self {
        Self {
            next: 1,
            samplers: BTreeMap::new(),
            candidates: BTreeMap::new(),
        }
    }

    pub(super) fn alloc_sampler(&mut self, node: SamplerNode) -> Result<u64> {
        if self.samplers.len() >= MAX_HANDLES {
            return Err(invalid("backend sampler handle bound"));
        }
        let id = self.next;
        self.next += 1;
        self.samplers.insert(id, node);
        Ok(id)
    }

    pub(super) fn get_sampler_mut(&mut self, id: u64) -> Result<&mut SamplerNode> {
        self.samplers
            .get_mut(&id)
            .ok_or_else(|| invalid("unknown backend sampler handle"))
    }

    pub(super) fn get_sampler(&self, id: u64) -> Result<&SamplerNode> {
        self.samplers
            .get(&id)
            .ok_or_else(|| invalid("unknown backend sampler handle"))
    }

    pub(super) fn init_sampler(&mut self, typ: &str, p: &Value) -> Result<Value> {
        let node = build_sampler_node(typ, p)?;
        let id = self.alloc_sampler(node)?;
        Ok(json!({ "Sampler": id }))
    }

    pub(super) fn chain_add(&mut self, chain_id: u64, child_id: u64) -> Result<Value> {
        if chain_id == child_id {
            return Err(invalid("backend sampler cannot own itself"));
        }
        let child = self
            .samplers
            .remove(&child_id)
            .ok_or_else(|| invalid("unknown child sampler"))?;
        let Some(SamplerNode::Chain(items)) = self.samplers.get_mut(&chain_id) else {
            return Err(invalid("target handle is not a sampler chain"));
        };
        items.push(child);
        Ok(Value::Null)
    }

    pub(super) fn clone_sampler(&mut self, id: u64) -> Result<Value> {
        let cloned = self.get_sampler(id)?.clone();
        let new_id = self.alloc_sampler(cloned)?;
        Ok(json!({ "Sampler": new_id }))
    }

    pub(super) fn copy_sampler(&mut self, dst_id: u64, src_id: u64) -> Result<Value> {
        let src = self.get_sampler(src_id)?.clone();
        *self.get_sampler_mut(dst_id)? = src;
        Ok(Value::Null)
    }

    pub(super) fn candidates_new(
        &mut self,
        operation: &Value,
        session_dir: &Path,
    ) -> Result<Value> {
        let count = operation["Count"]
            .as_u64()
            .and_then(|n| usize::try_from(n).ok())
            .filter(|&n| (1..=MAX_CANDIDATES).contains(&n))
            .ok_or_else(|| invalid("bs-candidates-new Count invalid"))?;
        let selected = operation["Selected"]
            .as_i64()
            .ok_or_else(|| invalid("bs-candidates-new Selected invalid"))?;
        let sorted = operation["Sorted"]
            .as_bool()
            .ok_or_else(|| invalid("bs-candidates-new Sorted invalid"))?;
        let file = checked_session_file(
            operation["File"]
                .as_str()
                .ok_or_else(|| invalid("bs-candidates-new File invalid"))?,
            session_dir,
            true,
        )?;
        let raw = fs::read(&file)?;
        if raw.len() != count * 12 {
            return Err(invalid("bs-candidates-new file extent differs"));
        }
        let mut backing = Vec::with_capacity(count);
        for chunk in raw.as_chunks::<12>().0 {
            let id = i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            let logit = f32::from_le_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]);
            let p = f32::from_le_bytes([chunk[8], chunk[9], chunk[10], chunk[11]]);
            backing.push(TokenCandidate { id, logit, p });
        }
        if self.candidates.len() >= MAX_HANDLES {
            return Err(invalid("bs-candidates handle bound"));
        }
        let id = self.next;
        self.next += 1;
        self.candidates.insert(
            id,
            CandidateBuffer {
                backing,
                offset: 0,
                size: count,
                selected,
                sorted,
            },
        );
        Ok(json!({ "Candidates": id }))
    }

    pub(super) fn candidates_apply(
        &mut self,
        sampler_id: u64,
        cand_id: u64,
        cb: &mut Option<SamplerEventCb<'_>>,
    ) -> Result<Value> {
        let mut sampler = self.get_sampler(sampler_id)?.clone();
        let cand = self
            .candidates
            .get_mut(&cand_id)
            .ok_or_else(|| invalid("unknown bs-candidates handle"))?;
        sampler.apply(cand, cb)?;
        let snap = cand.snapshot_json();
        *self.get_sampler_mut(sampler_id)? = sampler;
        Ok(snap)
    }

    pub(super) fn candidates_export(&self, operation: &Value, session_dir: &Path) -> Result<Value> {
        let cand_id = operation["Candidates"]
            .as_u64()
            .ok_or_else(|| invalid("bs-candidates-export Candidates missing"))?;
        let cand = self
            .candidates
            .get(&cand_id)
            .ok_or_else(|| invalid("unknown bs-candidates handle"))?;
        let selection = operation["Selection"]
            .as_str()
            .ok_or_else(|| invalid("bs-candidates-export Selection missing"))?;
        let (offset, count) = match selection {
            "active" => (cand.offset, cand.size),
            "prefix" => {
                let c = usize::try_from(
                    operation["Count"]
                        .as_u64()
                        .ok_or_else(|| invalid("prefix Count missing"))?,
                )?;
                if c > cand.backing.len() {
                    return Err(invalid("prefix Count exceeds backing"));
                }
                (0, c)
            }
            _ => return Err(invalid("backend sampler candidate selection invalid")),
        };
        let path = checked_session_file(
            operation["File"]
                .as_str()
                .ok_or_else(|| invalid("bs-candidates-export File missing"))?,
            session_dir,
            false,
        )?;
        let mut bytes = Vec::with_capacity(count * 12);
        for item in &cand.backing[offset..offset + count] {
            bytes.extend_from_slice(&item.id.to_le_bytes());
            bytes.extend_from_slice(&item.logit.to_le_bytes());
            bytes.extend_from_slice(&item.p.to_le_bytes());
        }
        fs::write(path, &bytes)?;
        let mut snap = cand.snapshot_json();
        snap["Rows"] = json!(count);
        snap["Bytes"] = json!(bytes.len());
        Ok(snap)
    }

    pub(super) fn candidates_free(&mut self, cand_id: u64) -> Result<Value> {
        self.candidates
            .remove(&cand_id)
            .ok_or_else(|| invalid("unknown bs-candidates handle"))?;
        Ok(Value::Null)
    }
}

fn bits_f32(val: &Value, key: &str) -> Result<f32> {
    let bits = u32::try_from(
        val.get(key)
            .and_then(Value::as_u64)
            .ok_or_else(|| invalid(format!("missing u32 bits {key}")))?,
    )?;
    let f = f32::from_bits(bits);
    if !f.is_finite() {
        return Err(invalid(format!("sampler parameter {key} must be finite")));
    }
    Ok(f)
}

pub(super) fn build_sampler_node(typ: &str, p: &Value) -> Result<SamplerNode> {
    match typ {
        "chain" => Ok(SamplerNode::Chain(Vec::new())),
        "greedy" => Ok(SamplerNode::Greedy),
        "dist" => Ok(SamplerNode::Dist(Mt19937::new(u32::try_from(
            p["Seed"]
                .as_u64()
                .ok_or_else(|| invalid("dist Seed invalid"))?,
        )?))),
        "temp" => Ok(SamplerNode::Temp(bits_f32(p, "TemperatureBits")?)),
        "temp-ext" => Ok(SamplerNode::TempExt {
            temp: bits_f32(p, "TemperatureBits")?,
            delta: bits_f32(p, "DeltaBits")?,
            exponent: bits_f32(p, "ExponentBits")?,
        }),
        "top-k" => Ok(SamplerNode::TopK(i32::try_from(
            p["K"].as_i64().ok_or_else(|| invalid("top-k K invalid"))?,
        )?)),
        "top-p" | "min-p" | "typical" => build_prob_sampler(typ, p),
        "xtc" => Ok(SamplerNode::Xtc {
            p: bits_f32(p, "ProbabilityBits")?,
            threshold: bits_f32(p, "ThresholdBits")?,
            min_keep: p["MinKeep"]
                .as_u64()
                .and_then(|n| usize::try_from(n).ok())
                .filter(|&n| n <= MAX_BACKEND_CANDIDATES)
                .ok_or_else(|| invalid("xtc MinKeep invalid"))?,
            rng: Mt19937::new(u32::try_from(
                p["Seed"]
                    .as_u64()
                    .ok_or_else(|| invalid("xtc Seed invalid"))?,
            )?),
        }),
        "logit-bias" => build_logit_bias(p),
        "penalties" => Ok(SamplerNode::Penalties {
            last_n: i32::try_from(
                p["PenaltyLastN"]
                    .as_i64()
                    .ok_or_else(|| invalid("PenaltyLastN"))?,
            )?,
            repeat: bits_f32(p, "RepeatBits")?,
            freq: bits_f32(p, "FrequencyBits")?,
            presence: bits_f32(p, "PresenceBits")?,
            history: Vec::new(),
        }),
        "dry-testing" => build_dry_sampler(p),
        "top-n-sigma" => Ok(SamplerNode::TopNSigma(bits_f32(p, "NBits")?)),
        "observer" => Ok(SamplerNode::Observer {
            label: u32::try_from(
                p["Label"]
                    .as_u64()
                    .ok_or_else(|| invalid("observer Label missing"))?,
            )?,
        }),
        _ => Err(invalid(format!("unknown sampler type {typ}"))),
    }
}

fn build_prob_sampler(typ: &str, p: &Value) -> Result<SamplerNode> {
    let prob = bits_f32(p, "ProbabilityBits")?;
    let min_keep = p["MinKeep"]
        .as_u64()
        .and_then(|n| usize::try_from(n).ok())
        .filter(|&n| n <= MAX_BACKEND_CANDIDATES)
        .ok_or_else(|| invalid("sampler MinKeep invalid"))?;
    Ok(match typ {
        "top-p" => SamplerNode::TopP { p: prob, min_keep },
        "min-p" => SamplerNode::MinP { p: prob, min_keep },
        _ => SamplerNode::Typical { p: prob, min_keep },
    })
}

fn build_logit_bias(p: &Value) -> Result<SamplerNode> {
    let arr = p["Biases"]
        .as_array()
        .ok_or_else(|| invalid("Biases must be array"))?;
    let mut biases = Vec::with_capacity(arr.len());
    for item in arr {
        let tok = i32::try_from(
            item["Token"]
                .as_i64()
                .ok_or_else(|| invalid("bias Token missing"))?,
        )?;
        biases.push((tok, bits_f32(item, "BiasBits")?));
    }
    Ok(SamplerNode::LogitBias(biases))
}

fn build_dry_sampler(p: &Value) -> Result<SamplerNode> {
    let mult = bits_f32(p, "MultiplierBits")?;
    let base = bits_f32(p, "BaseBits")?;
    let allowed_len = i32::try_from(
        p["AllowedLength"]
            .as_i64()
            .ok_or_else(|| invalid("AllowedLength"))?,
    )?;
    let last_n = i32::try_from(
        p["PenaltyLastN"]
            .as_i64()
            .ok_or_else(|| invalid("PenaltyLastN"))?,
    )?;
    let arr = p["SequenceBreakers"]
        .as_array()
        .ok_or_else(|| invalid("SequenceBreakers must be array"))?;
    let mut breakers = Vec::with_capacity(arr.len());
    for b in arr {
        let inner = b
            .as_array()
            .ok_or_else(|| invalid("breaker must be array"))?;
        let mut toks = Vec::with_capacity(inner.len());
        for t in inner {
            toks.push(i32::try_from(
                t.as_i64().ok_or_else(|| invalid("breaker tok"))?,
            )?);
        }
        breakers.push(toks);
    }
    Ok(SamplerNode::Dry {
        mult,
        base,
        allowed_len,
        last_n,
        breakers,
        history: Vec::new(),
    })
}

pub(super) fn bs_tokenize(operation: &Value, model: &ModelData) -> Result<Value> {
    let hex_str = operation["InputHex"]
        .as_str()
        .ok_or_else(|| invalid("bs-tokenize InputHex invalid"))?;
    let input_bytes = hex::decode(hex_str)?;
    let capacity = operation["Capacity"]
        .as_u64()
        .and_then(|n| usize::try_from(n).ok())
        .filter(|&n| n <= MAX_CANDIDATES)
        .ok_or_else(|| invalid("bs-tokenize Capacity invalid"))?;
    let add_special = operation["AddSpecial"]
        .as_bool()
        .ok_or_else(|| invalid("bs-tokenize AddSpecial invalid"))?;
    let parse_special = operation["ParseSpecial"]
        .as_bool()
        .ok_or_else(|| invalid("bs-tokenize ParseSpecial invalid"))?;
    let text = String::from_utf8_lossy(&input_bytes);
    let toks = model.vocab.tokenize(&text, add_special, parse_special)?;
    let mut backing = vec![0_i32; capacity];
    let ret = if toks.len() <= capacity {
        backing[..toks.len()].copy_from_slice(&toks);
        i32::try_from(toks.len())?
    } else {
        -i32::try_from(toks.len())?
    };
    Ok(json!({ "ReturnCode": ret, "Backing": backing }))
}

pub(super) fn bs_piece(operation: &Value, model: &ModelData) -> Result<Value> {
    let token = i32::try_from(
        operation["Token"]
            .as_i64()
            .ok_or_else(|| invalid("bs-piece Token invalid"))?,
    )?;
    let capacity = operation["Capacity"]
        .as_u64()
        .and_then(|n| usize::try_from(n).ok())
        .filter(|&n| n <= MAX_CANDIDATES)
        .ok_or_else(|| invalid("bs-piece Capacity invalid"))?;
    let lstrip = i32::try_from(
        operation["LStrip"]
            .as_i64()
            .ok_or_else(|| invalid("bs-piece LStrip invalid"))?,
    )?;
    let special = operation["Special"]
        .as_bool()
        .ok_or_else(|| invalid("bs-piece Special invalid"))?;
    let piece = model.vocab.token_to_piece(token, lstrip, special)?;
    let mut backing = vec![0_u8; capacity];
    let ret = if piece.len() <= capacity {
        backing[..piece.len()].copy_from_slice(&piece);
        i32::try_from(piece.len())?
    } else {
        -i32::try_from(piece.len())?
    };
    Ok(json!({
        "ReturnCode": ret,
        "BackingHex": hex::encode(&backing),
    }))
}
