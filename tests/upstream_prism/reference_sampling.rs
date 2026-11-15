//! Clean-room independent safe Rust sampler and candidate-view reference provider.
//!
//! No production `bonsai_prism_runtime` code, native bridge or precomputed
//! answer fixture is called here. Implements stateful MSVC MT19937 seeding,
//! backing-preserving candidate views, libstdc++ `partial_sort` tie-breaking,
//! repetition history and all eleven `sampler-trace` operations directly.

#![expect(
    clippy::float_cmp,
    clippy::suboptimal_flops,
    reason = "Preserve native binary32/binary64 operation roundings and exact identity guards."
)]

use std::cmp::Ordering;
use std::collections::{BTreeMap, VecDeque};
use std::hash::{BuildHasher, Hasher};
use std::time::Instant;

use serde_json::{Value, json};

use super::{Result, invalid};

pub(super) const ID: &str = "tests/test-sampling.cpp";
const SOURCE_SHA256: &str = "817543792852df22e78f705192793ec5797c872708274d572c24c2fb0c95a8ac";
const MAX_CANDIDATES: usize = 131_072;
const MAX_RETAINED: usize = 262_144;
const MAX_HANDLE: u32 = 31;

#[derive(Clone, Copy)]
struct Candidate {
    id: i32,
    logit: f32,
    probability: f32,
}

#[derive(Clone)]
struct CandidateView {
    backing: Vec<Candidate>,
    offset: usize,
    size: usize,
    selected: i64,
    sorted: bool,
}

impl CandidateView {
    fn new(backing: Vec<Candidate>, selected: i64, sorted: bool) -> Self {
        Self {
            size: backing.len(),
            backing,
            offset: 0,
            selected,
            sorted,
        }
    }

    fn active(&self) -> &[Candidate] {
        &self.backing[self.offset..self.offset + self.size]
    }

    fn active_mut(&mut self) -> &mut [Candidate] {
        &mut self.backing[self.offset..self.offset + self.size]
    }

    const fn validate(&self) -> bool {
        self.offset <= self.backing.len() && self.size <= self.backing.len() - self.offset
    }

    fn sort_descending(&mut self) {
        self.active_mut().sort_unstable_by(|left, right| {
            right
                .logit
                .partial_cmp(&left.logit)
                .unwrap_or(Ordering::Equal)
                .then_with(|| left.id.cmp(&right.id))
        });
        self.sorted = true;
    }

    fn softmax(&mut self, sort_first: bool) {
        if sort_first && !self.sorted {
            self.sort_descending();
        }
        let mut maximum = self.active()[0].logit;
        if !self.sorted {
            for entry in &self.active()[1..] {
                if maximum < entry.logit {
                    maximum = entry.logit;
                }
            }
        }
        if !maximum.is_finite() {
            let mut sum = 0.0_f32;
            for entry in self.active_mut() {
                entry.probability = if entry.logit.to_bits() == maximum.to_bits() {
                    1.0
                } else {
                    0.0
                };
                sum += entry.probability;
            }
            for entry in self.active_mut() {
                entry.probability /= sum;
            }
            return;
        }
        let mut sum = 0.0_f32;
        for entry in self.active_mut() {
            entry.probability = (entry.logit - maximum).exp();
            sum += entry.probability;
        }
        for entry in self.active_mut() {
            entry.probability /= sum;
        }
    }
}

#[derive(Clone)]
struct Mt19937 {
    state: [u32; 624],
    index: usize,
}

impl Mt19937 {
    fn new(seed: u32) -> Self {
        let seed = if seed == u32::MAX {
            let mut hasher = std::collections::hash_map::RandomState::new().build_hasher();
            hasher.write_u32(std::process::id());
            let bytes = hasher.finish().to_le_bytes();
            u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
        } else {
            seed
        };
        let mut state = [0_u32; 624];
        state[0] = seed;
        for (idx, inc) in (1..624).zip(1_u32..) {
            state[idx] = 1_812_433_253_u32
                .wrapping_mul(state[idx - 1] ^ (state[idx - 1] >> 30))
                .wrapping_add(inc);
        }
        Self { state, index: 624 }
    }

    fn next_u32(&mut self) -> u32 {
        if self.index == 624 {
            for idx in 0..624 {
                let mixed =
                    (self.state[idx] & 0x8000_0000) | (self.state[(idx + 1) % 624] & 0x7fff_ffff);
                self.state[idx] = self.state[(idx + 397) % 624]
                    ^ (mixed >> 1)
                    ^ if mixed & 1 == 0 { 0 } else { 0x9908_b0df };
            }
            self.index = 0;
        }
        let mut y = self.state[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^ (y >> 18)
    }

    #[expect(
        clippy::as_conversions,
        clippy::cast_precision_loss,
        reason = "Canonical binary32 conversion from exact 24-bit MT19937 quotient."
    )]
    fn unit_f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 * (1.0 / 16_777_216.0)
    }

    #[expect(
        clippy::as_conversions,
        clippy::cast_precision_loss,
        reason = "MSVC canonical binary64 combines 21 + 32 MT19937 bits into an exact 53-bit integer."
    )]
    fn unit_f64(&mut self) -> f64 {
        let low = u64::from(self.next_u32() >> 11);
        let high = u64::from(self.next_u32()) << 21;
        (low + high) as f64 * (1.0 / 9_007_199_254_740_992.0)
    }
}

#[derive(Clone)]
enum SamplerSpec {
    Temp(f32),
    TempExt {
        temperature: f32,
        delta: f32,
        exponent: f32,
    },
    Dist(u32),
    TopK(i32),
    TopP {
        probability: f32,
        min_keep: usize,
    },
    MinP {
        probability: f32,
        min_keep: usize,
    },
    Typical {
        probability: f32,
        min_keep: usize,
    },
    Xtc {
        probability: f32,
        threshold: f32,
        min_keep: usize,
        seed: u32,
    },
    Penalties {
        last_n: i32,
        repeat: f32,
        frequency: f32,
        presence: f32,
    },
    Dry {
        multiplier: f32,
        base: f32,
        allowed_length: usize,
        last_n: i32,
        breakers: Vec<Vec<i32>>,
    },
    TopNSigma(f32),
}

impl SamplerSpec {
    const fn seed(&self) -> Option<u32> {
        match self {
            Self::Dist(seed) | Self::Xtc { seed, .. } => Some(*seed),
            _ => None,
        }
    }

    fn history_limit(&self) -> usize {
        match self {
            Self::Penalties { last_n, .. } | Self::Dry { last_n, .. } => {
                usize::try_from(*last_n).unwrap_or(0)
            }
            _ => 0,
        }
    }
}

struct Sampler {
    spec: SamplerSpec,
    rng: Option<Mt19937>,
    history: VecDeque<i32>,
}

impl Sampler {
    fn new(spec: SamplerSpec) -> Self {
        let rng = spec.seed().map(Mt19937::new);
        Self {
            spec,
            rng,
            history: VecDeque::new(),
        }
    }

    fn reset(&mut self) {
        self.history.clear();
        self.rng = self.spec.seed().map(Mt19937::new);
    }

    fn accept(&mut self, token: i32) {
        let cap = self.spec.history_limit();
        if cap > 0 {
            if self.history.len() == cap {
                self.history.pop_front();
            }
            self.history.push_back(token);
        }
    }

    fn apply(&mut self, view: &mut CandidateView) -> Result<()> {
        if view.active().is_empty() {
            return Err(invalid("cannot apply sampler to empty candidate view"));
        }
        match &self.spec {
            SamplerSpec::Temp(temp) => apply_temp(view, *temp),
            SamplerSpec::TempExt {
                temperature,
                delta,
                exponent,
            } => apply_temp_ext(view, *temperature, *delta, *exponent),
            SamplerSpec::Dist(_) => {
                let rng = self
                    .rng
                    .as_mut()
                    .ok_or_else(|| invalid("missing dist RNG state"))?;
                apply_dist(view, rng);
            }
            SamplerSpec::TopK(k) => apply_top_k(view, *k)?,
            SamplerSpec::TopP {
                probability,
                min_keep,
            } => apply_top_p(view, *probability, *min_keep),
            SamplerSpec::MinP {
                probability,
                min_keep,
            } => apply_min_p(view, *probability, *min_keep),
            SamplerSpec::Typical {
                probability,
                min_keep,
            } => apply_typical(view, *probability, *min_keep),
            SamplerSpec::Xtc {
                probability,
                threshold,
                min_keep,
                ..
            } => {
                let rng = self
                    .rng
                    .as_mut()
                    .ok_or_else(|| invalid("missing xtc RNG state"))?;
                apply_xtc(view, *probability, *threshold, *min_keep, rng);
            }
            SamplerSpec::Penalties {
                last_n,
                repeat,
                frequency,
                presence,
            } => {
                if *last_n > 0 && (*repeat != 1.0 || *frequency != 0.0 || *presence != 0.0) {
                    apply_penalties(view, &self.history, *repeat, *frequency, *presence);
                }
            }
            SamplerSpec::Dry {
                multiplier,
                base,
                allowed_length,
                last_n,
                breakers,
            } => {
                if *last_n > 0 && *multiplier != 0.0 && *base >= 1.0 {
                    apply_dry(
                        view,
                        &self.history,
                        *multiplier,
                        *base,
                        *allowed_length,
                        breakers,
                    );
                }
            }
            SamplerSpec::TopNSigma(n) => apply_top_n_sigma(view, *n),
        }
        Ok(())
    }
}

#[expect(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "Overflow-guarded logit temperature division narrows double differences into binary32."
)]
fn apply_temp(view: &mut CandidateView, temperature: f32) {
    if temperature <= 0.0 {
        let mut best = 0;
        for idx in 1..view.size {
            if view.active()[idx].logit > view.active()[best].logit {
                view.active_mut()[best].logit = f32::NEG_INFINITY;
                best = idx;
            } else {
                view.active_mut()[idx].logit = f32::NEG_INFINITY;
            }
        }
        return;
    }
    if temperature.to_bits() == 1.0_f32.to_bits() {
        return;
    }
    let mut maximum = f32::NEG_INFINITY;
    let mut overflows = false;
    let max_f32 = f64::from(f32::MAX);
    let temp_f64 = f64::from(temperature);
    for entry in view.active() {
        if entry.logit > maximum {
            maximum = entry.logit;
        }
        if entry.logit.is_finite() {
            let scaled = f64::from(entry.logit) / temp_f64;
            if !scaled.is_finite() || scaled.abs() > max_f32 {
                overflows = true;
            }
        }
    }
    if overflows && maximum.is_finite() {
        let max_f64 = f64::from(maximum);
        for entry in view.active_mut() {
            if entry.logit.is_finite() {
                entry.logit = ((f64::from(entry.logit) - max_f64) / temp_f64) as f32;
            }
        }
    } else {
        for entry in view.active_mut() {
            entry.logit /= temperature;
        }
    }
}

#[expect(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "Match native binary32 active-size scaling and binary64-to-binary32 probability store."
)]
fn apply_temp_ext(view: &mut CandidateView, temperature: f32, delta: f32, exponent: f32) {
    if delta <= 0.0 {
        apply_temp(view, temperature);
        return;
    }
    if view.size <= 1 {
        return;
    }
    let min_temp = 0.0_f32.max(temperature - delta);
    let max_temp = temperature + delta;
    let max_entropy = -(1.0_f32 / view.size as f32).ln();
    view.softmax(true);
    let mut entropy = 0.0_f32;
    for entry in view.active() {
        if entry.probability > 0.0 {
            entropy -= entry.probability * entry.probability.ln();
        }
    }
    let dyn_temp = min_temp + (max_temp - min_temp) * (entropy / max_entropy).powf(exponent);
    apply_temp(view, dyn_temp);
    let max_logit = view.active()[0].logit;
    let mut sum = 0.0_f64;
    if max_logit.is_finite() {
        let max_f64 = f64::from(max_logit);
        for entry in view.active_mut() {
            let w = (f64::from(entry.logit) - max_f64).exp();
            entry.probability = w as f32;
            sum += w;
        }
    } else {
        for entry in view.active_mut() {
            let w = if entry.logit.to_bits() == max_logit.to_bits() {
                1.0
            } else {
                0.0
            };
            entry.probability = w as f32;
            sum += w;
        }
    }
    for entry in view.active_mut() {
        entry.probability = (f64::from(entry.probability) / sum) as f32;
    }
}

#[expect(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "Binary64 cumulative normalization stores into binary32 probability."
)]
fn apply_dist(view: &mut CandidateView, rng: &mut Mt19937) {
    view.selected = 0;
    let draw = rng.unit_f64();
    if view.size == 1 {
        view.active_mut()[0].probability = 1.0;
        return;
    }
    let mut maximum = view.active()[0].logit;
    if !view.sorted {
        for entry in &view.active()[1..] {
            if maximum < entry.logit {
                maximum = entry.logit;
            }
        }
    }
    let mut sum = 0.0_f64;
    if maximum.is_finite() {
        for entry in view.active_mut() {
            entry.probability = (entry.logit - maximum).exp();
            sum += f64::from(entry.probability);
        }
    } else {
        for entry in view.active_mut() {
            let w = if entry.logit.to_bits() == maximum.to_bits() {
                1.0
            } else {
                0.0
            };
            entry.probability = w as f32;
            sum += w;
        }
    }
    let target = sum * draw;
    let mut cumulative = 0.0_f64;
    let mut picked = None;
    for (idx, entry) in view.active_mut().iter_mut().enumerate() {
        if picked.is_none() {
            cumulative += f64::from(entry.probability);
            if cumulative >= target {
                picked = Some(idx);
            }
        }
        entry.probability = (f64::from(entry.probability) / sum) as f32;
    }
    view.selected = i64::try_from(picked.unwrap_or(view.size - 1)).unwrap_or(-1);
}

fn adjust_min_heap(data: &mut [Candidate], mut hole: usize, length: usize, value: Candidate) {
    let top = hole;
    let mut child = hole;
    while child < (length - 1) / 2 {
        child = 2 * (child + 1);
        if data[child].logit > data[child - 1].logit {
            child -= 1;
        }
        data[hole] = data[child];
        hole = child;
    }
    if length.is_multiple_of(2) && child == (length - 2) / 2 {
        child = 2 * (child + 1);
        data[hole] = data[child - 1];
        hole = child - 1;
    }
    while hole > top {
        let parent = (hole - 1) / 2;
        if data[parent].logit.partial_cmp(&value.logit) != Some(Ordering::Greater) {
            break;
        }
        data[hole] = data[parent];
        hole = parent;
    }
    data[hole] = value;
}

fn partial_top_k(data: &mut [Candidate], keep: usize) {
    if keep > 1 {
        for parent in (0..=(keep - 2) / 2).rev() {
            let value = data[parent];
            adjust_min_heap(data, parent, keep, value);
        }
    }
    for idx in keep..data.len() {
        if data[idx].logit > data[0].logit {
            let value = data[idx];
            data[idx] = data[0];
            adjust_min_heap(data, 0, keep, value);
        }
    }
    for end in (1..keep).rev() {
        let value = data[end];
        data[end] = data[0];
        adjust_min_heap(data, 0, end, value);
    }
}

fn apply_top_k(view: &mut CandidateView, k: i32) -> Result<()> {
    if k <= 0 {
        return Ok(());
    }
    let keep = view.size.min(usize::try_from(k)?);
    if !view.sorted {
        if keep <= 128 {
            partial_top_k(view.active_mut(), keep);
            view.sorted = true;
        } else {
            view.sort_descending();
        }
    }
    view.size = keep;
    Ok(())
}

fn apply_top_p(view: &mut CandidateView, probability: f32, min_keep: usize) {
    if probability >= 1.0 {
        return;
    }
    view.softmax(false);
    if !view.sorted {
        view.sort_descending();
    }
    let mut cumulative = 0.0_f32;
    let mut next_size = view.size;
    for (idx, entry) in view.active().iter().enumerate() {
        cumulative += entry.probability;
        if cumulative >= probability && idx + 1 >= min_keep {
            next_size = idx + 1;
            break;
        }
    }
    view.size = next_size;
}

fn apply_min_p(view: &mut CandidateView, probability: f32, min_keep: usize) {
    if probability <= 0.0 {
        return;
    }
    let threshold = f64::from(probability).ln();
    if !view.sorted {
        let mut maximum = -f32::MAX;
        for entry in view.active() {
            if maximum < entry.logit {
                maximum = entry.logit;
            }
        }
        let max_f64 = f64::from(maximum);
        let filtered: Vec<Candidate> = view
            .active()
            .iter()
            .filter(|entry| {
                entry.logit.to_bits() == maximum.to_bits()
                    || f64::from(entry.logit) - max_f64 >= threshold
            })
            .copied()
            .collect();
        if !filtered.is_empty() && filtered.len() >= min_keep {
            view.active_mut()[..filtered.len()].copy_from_slice(&filtered);
            view.size = filtered.len();
            return;
        }
        view.sort_descending();
    }
    let maximum = view.active()[0].logit;
    let max_f64 = f64::from(maximum);
    let mut size = 1;
    while size < view.size {
        let logit = view.active()[size].logit;
        if logit.to_bits() != maximum.to_bits()
            && f64::from(logit) - max_f64 < threshold
            && size >= min_keep
        {
            break;
        }
        size += 1;
    }
    view.size = size;
}

fn apply_typical(view: &mut CandidateView, probability: f32, min_keep: usize) {
    if probability >= 1.0 {
        return;
    }
    view.softmax(true);
    let mut entropy = 0.0_f32;
    for entry in view.active() {
        entropy += -entry.probability * entry.probability.ln();
    }
    let scores: Vec<f32> = view
        .active()
        .iter()
        .map(|entry| (-entry.probability.ln() - entropy).abs())
        .collect();
    let mut indices: Vec<usize> = (0..view.size).collect();
    indices.sort_unstable_by(|&a, &b| scores[a].partial_cmp(&scores[b]).unwrap_or(Ordering::Equal));
    let mut cumulative = 0.0_f32;
    let mut keep = indices.len();
    for (idx, &pos) in indices.iter().enumerate() {
        cumulative += view.active()[pos].probability;
        if cumulative > probability && idx + 1 >= min_keep {
            keep = idx + 1;
            break;
        }
    }
    let filtered: Vec<Candidate> = indices[..keep]
        .iter()
        .map(|&pos| view.active()[pos])
        .collect();
    view.active_mut()[..keep].copy_from_slice(&filtered);
    view.size = keep;
    view.sorted = false;
}

fn apply_xtc(
    view: &mut CandidateView,
    probability: f32,
    threshold: f32,
    min_keep: usize,
    rng: &mut Mt19937,
) {
    if probability <= 0.0 || threshold > 0.5 || view.size < 2 || rng.unit_f32() > probability {
        return;
    }
    view.softmax(true);
    let mut last = 0;
    for (idx, entry) in view.active().iter().enumerate() {
        if entry.probability >= threshold {
            last = idx;
        } else {
            break;
        }
    }
    if last > 0 && view.size - last >= min_keep {
        view.offset += last;
        view.size -= last;
    }
}

#[expect(
    clippy::as_conversions,
    clippy::cast_precision_loss,
    reason = "Bounded token occurrence count converts directly to binary32 for penalty subtraction."
)]
fn apply_penalties(
    view: &mut CandidateView,
    history: &VecDeque<i32>,
    repeat: f32,
    frequency: f32,
    presence: f32,
) {
    let mut counts = BTreeMap::<i32, usize>::new();
    for &tok in history {
        *counts.entry(tok).or_default() += 1;
    }
    for entry in view.active_mut() {
        if let Some(&count) = counts.get(&entry.id) {
            if entry.logit <= 0.0 {
                entry.logit *= repeat;
            } else {
                entry.logit /= repeat;
            }
            entry.logit -= count as f32 * frequency + presence;
        }
    }
    view.sorted = false;
}

fn dry_limit(reversed: &[i32], breakers: &[Vec<i32>]) -> usize {
    for (idx, tok) in reversed.iter().enumerate() {
        let mut longest = None;
        for breaker in breakers {
            if breaker.first() == Some(tok) {
                let tail = &breaker[1..];
                if tail.len() <= idx
                    && tail
                        .iter()
                        .enumerate()
                        .all(|(off, &t)| t == reversed[idx - off - 1])
                {
                    longest = Some(longest.unwrap_or(0).max(tail.len()));
                }
            }
        }
        if let Some(len) = longest {
            return idx - len;
        }
    }
    reversed.len()
}

fn dry_repeat_lengths(reversed: &[i32], limit: usize) -> Vec<usize> {
    let mut lengths = vec![0_usize; reversed.len()];
    let last = reversed.len() - 1;
    let mut left = 0;
    let mut right = 0;
    for offset in 1..reversed.len() {
        if offset > right {
            let mut len = 0;
            while len + offset < reversed.len() && reversed[len] == reversed[len + offset] {
                len += 1;
            }
            lengths[last - offset] = len.min(limit);
            if len > 0 {
                left = offset;
                right = offset + len - 1;
            }
        } else {
            let pair = offset - left;
            let remaining = right - offset + 1;
            if lengths[last - pair] < remaining {
                lengths[last - offset] = lengths[last - pair].min(limit);
            } else {
                let mut idx = right + 1;
                while idx < reversed.len() && reversed[idx] == reversed[idx - offset] {
                    idx += 1;
                }
                lengths[last - offset] = (idx - offset).min(limit);
                left = offset;
                right = idx - 1;
            }
        }
    }
    lengths
}

#[expect(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "Bounded logarithmic exponent cap matches native DRY float-to-size clamp."
)]
fn apply_dry(
    view: &mut CandidateView,
    history: &VecDeque<i32>,
    multiplier: f32,
    base: f32,
    allowed: usize,
    breakers: &[Vec<i32>],
) {
    if history.len() <= allowed {
        return;
    }
    let reversed: Vec<i32> = history.iter().rev().copied().collect();
    let limit = dry_limit(&reversed, breakers);
    if limit < allowed {
        return;
    }
    let lengths = dry_repeat_lengths(&reversed, limit);
    let mut counts = BTreeMap::<i32, usize>::new();
    for (idx, &len) in lengths[..lengths.len() - 1].iter().enumerate() {
        if len >= allowed {
            let tok = reversed[lengths.len() - 2 - idx];
            let entry = counts.entry(tok).or_default();
            *entry = (*entry).max(len);
        }
    }
    let max_exp = if base > 1.000_001 {
        (88.722_84_f32 / base.ln()) as usize
    } else {
        0
    };
    for entry in view.active_mut() {
        if let Some(&rep) = counts.get(&entry.id) {
            if breakers.iter().any(|b| b.len() == 1 && b[0] == entry.id) {
                continue;
            }
            let mut exp = rep - allowed;
            if max_exp > 0 {
                exp = exp.min(max_exp);
            }
            let penalty = (f64::from(multiplier)
                * f64::from(base).powi(i32::try_from(exp).unwrap_or(i32::MAX)))
                as f32;
            entry.logit -= penalty;
        }
    }
    view.sorted = false;
}

fn apply_top_n_sigma(view: &mut CandidateView, n: f32) {
    if n <= 0.0 || view.size <= 1 {
        return;
    }
    let mut maximum = view.active()[0].logit;
    let mut sum = 0.0_f64;
    let mut count = 0_u32;
    for entry in view.active() {
        if entry.logit != f32::NEG_INFINITY {
            if maximum < entry.logit {
                maximum = entry.logit;
            }
            sum += f64::from(entry.logit);
            count += 1;
        }
    }
    let count_f64 = f64::from(count);
    let mean = if count > 0 { sum / count_f64 } else { 0.0 };
    let mut variance = 0.0_f64;
    for entry in view.active() {
        if entry.logit != f32::NEG_INFINITY {
            let diff = f64::from(entry.logit) - mean;
            variance += diff * diff;
        }
    }
    let stddev = if count > 0 {
        (variance / count_f64).sqrt()
    } else {
        0.0
    };
    let threshold = f64::from(maximum) - f64::from(n) * stddev;
    for entry in view.active_mut() {
        if f64::from(entry.logit) < threshold {
            entry.logit = f32::NEG_INFINITY;
        }
    }
}

fn parse_sampler_spec(kind: &str, params: &Value) -> Result<SamplerSpec> {
    Ok(match kind {
        "temp" => {
            fields(params, &["TemperatureBits"])?;
            SamplerSpec::Temp(finite_float(&params["TemperatureBits"])?)
        }
        "temp-ext" => {
            fields(params, &["TemperatureBits", "DeltaBits", "ExponentBits"])?;
            SamplerSpec::TempExt {
                temperature: finite_float(&params["TemperatureBits"])?,
                delta: finite_float(&params["DeltaBits"])?,
                exponent: finite_float(&params["ExponentBits"])?,
            }
        }
        "dist" => {
            fields(params, &["Seed"])?;
            SamplerSpec::Dist(u32_bound(&params["Seed"], u32::MAX)?)
        }
        "top-k" => {
            fields(params, &["K"])?;
            SamplerSpec::TopK(i32_val(&params["K"])?)
        }
        "top-p" | "min-p" | "typical" => {
            fields(params, &["ProbabilityBits", "MinKeep"])?;
            let probability = finite_float(&params["ProbabilityBits"])?;
            let min_keep = usize::try_from(u32_bound(
                &params["MinKeep"],
                u32::try_from(MAX_CANDIDATES)?,
            )?)?;
            match kind {
                "top-p" => SamplerSpec::TopP {
                    probability,
                    min_keep,
                },
                "min-p" => SamplerSpec::MinP {
                    probability,
                    min_keep,
                },
                _ => SamplerSpec::Typical {
                    probability,
                    min_keep,
                },
            }
        }
        "xtc" => {
            fields(
                params,
                &["ProbabilityBits", "ThresholdBits", "MinKeep", "Seed"],
            )?;
            SamplerSpec::Xtc {
                probability: finite_float(&params["ProbabilityBits"])?,
                threshold: finite_float(&params["ThresholdBits"])?,
                min_keep: usize::try_from(u32_bound(
                    &params["MinKeep"],
                    u32::try_from(MAX_CANDIDATES)?,
                )?)?,
                seed: u32_bound(&params["Seed"], u32::MAX)?,
            }
        }
        "penalties" => {
            fields(
                params,
                &[
                    "VocabularySize",
                    "PenaltyLastN",
                    "RepeatBits",
                    "FrequencyBits",
                    "PresenceBits",
                ],
            )?;
            if u32_bound(&params["VocabularySize"], u32::try_from(MAX_CANDIDATES)?)? == 0 {
                return Err(invalid("sampler vocabulary cannot be empty"));
            }
            let last_n = i32_val(&params["PenaltyLastN"])?;
            if !(-1..=i32::try_from(MAX_CANDIDATES)?).contains(&last_n) {
                return Err(invalid("penalty history exceeds bound"));
            }
            let repeat = finite_float(&params["RepeatBits"])?;
            if repeat <= 0.0 {
                return Err(invalid("repeat penalty must be positive"));
            }
            SamplerSpec::Penalties {
                last_n,
                repeat,
                frequency: finite_float(&params["FrequencyBits"])?,
                presence: finite_float(&params["PresenceBits"])?,
            }
        }
        "dry-testing" => parse_dry_spec(params)?,
        "top-n-sigma" => {
            fields(params, &["NBits"])?;
            SamplerSpec::TopNSigma(finite_float(&params["NBits"])?)
        }
        _ => return Err(invalid("unknown sampler Type")),
    })
}

fn parse_dry_spec(params: &Value) -> Result<SamplerSpec> {
    fields(
        params,
        &[
            "MultiplierBits",
            "BaseBits",
            "AllowedLength",
            "PenaltyLastN",
            "SequenceBreakers",
        ],
    )?;
    let allowed = i32_val(&params["AllowedLength"])?;
    let last_n = i32_val(&params["PenaltyLastN"])?;
    if !(0..=i32::try_from(MAX_CANDIDATES)?).contains(&allowed)
        || !(-1..=i32::try_from(MAX_CANDIDATES)?).contains(&last_n)
    {
        return Err(invalid("DRY parameter outside bound"));
    }
    let breaker_values = array(&params["SequenceBreakers"])?;
    if breaker_values.len() > 128 {
        return Err(invalid("too many DRY sequence breakers"));
    }
    let mut breakers = Vec::with_capacity(breaker_values.len());
    let mut token_sum = 0_usize;
    for entry in breaker_values {
        let items = array(entry)?;
        if items.len() > 1024 {
            return Err(invalid("DRY sequence breaker exceeds bound"));
        }
        token_sum += items.len();
        if token_sum > 8192 {
            return Err(invalid("DRY sequence breaker total tokens exceed bound"));
        }
        breakers.push(items.iter().map(i32_val).collect::<Result<Vec<_>>>()?);
    }
    Ok(SamplerSpec::Dry {
        multiplier: finite_float(&params["MultiplierBits"])?,
        base: finite_float(&params["BaseBits"])?,
        allowed_length: usize::try_from(allowed)?,
        last_n,
        breakers,
    })
}

fn parse_candidates(action: &Value) -> Result<CandidateView> {
    let entries = array(&action["Data"])?;
    if entries.is_empty() || entries.len() > MAX_CANDIDATES {
        return Err(invalid("candidate vector extent outside bound"));
    }
    let selected = i64_val(&action["Selected"])?;
    if selected < -1 || (selected >= 0 && u64::try_from(selected)? >= u64::try_from(entries.len())?)
    {
        return Err(invalid("selected candidate outside extent"));
    }
    let sorted = action["Sorted"]
        .as_bool()
        .ok_or_else(|| invalid("candidate Sorted must be boolean"))?;
    let mut backing = Vec::with_capacity(entries.len());
    for entry in entries {
        fields(entry, &["Id", "LogitBits", "ProbabilityBits"])?;
        backing.push(Candidate {
            id: i32_val(&entry["Id"])?,
            logit: raw_float(&entry["LogitBits"])?,
            probability: raw_float(&entry["ProbabilityBits"])?,
        });
    }
    Ok(CandidateView::new(backing, selected, sorted))
}

fn snapshot_candidates(view: &CandidateView) -> Result<Value> {
    if !view.validate() {
        return Err(invalid("candidate view exceeds backing storage"));
    }
    let data = view
        .active()
        .iter()
        .map(|entry| {
            json!({
                "Id": entry.id,
                "LogitBits": entry.logit.to_bits(),
                "ProbabilityBits": entry.probability.to_bits(),
            })
        })
        .collect::<Vec<_>>();
    Ok(json!({
        "Data": data,
        "Size": view.size,
        "Selected": view.selected,
        "Sorted": view.sorted,
    }))
}

fn execute_trace_action(
    action: &Value,
    clock: &Instant,
    samplers: &mut BTreeMap<u32, Sampler>,
    candidates: &mut BTreeMap<u32, CandidateView>,
) -> Result<Value> {
    let mut event = json!({});
    match text(&action["Kind"])? {
        "clock" => {
            fields(action, &["Kind"])?;
            event = json!({"Microseconds": i64::try_from(clock.elapsed().as_micros())?});
        }
        "init-sampler" => {
            fields(action, &["Kind", "Sampler", "Type", "Parameters"])?;
            let id = u32_bound(&action["Sampler"], MAX_HANDLE)?;
            if samplers.contains_key(&id) {
                return Err(invalid("sampler handle already exists"));
            }
            let spec = parse_sampler_spec(text(&action["Type"])?, &action["Parameters"])?;
            samplers.insert(id, Sampler::new(spec));
        }
        "init-candidates" => {
            fields(
                action,
                &["Kind", "Candidates", "Data", "Selected", "Sorted"],
            )?;
            let id = u32_bound(&action["Candidates"], MAX_HANDLE)?;
            if candidates.contains_key(&id) {
                return Err(invalid("candidate handle already exists"));
            }
            candidates.insert(id, parse_candidates(action)?);
        }
        "copy-candidates" => {
            fields(action, &["Kind", "Source", "Target"])?;
            let src = u32_bound(&action["Source"], MAX_HANDLE)?;
            let dst = u32_bound(&action["Target"], MAX_HANDLE)?;
            let copied = candidates
                .get(&src)
                .ok_or_else(|| invalid("unknown candidate source"))?
                .clone();
            if src != dst {
                candidates.insert(dst, copied);
            }
        }
        "free-candidates" => {
            fields(action, &["Kind", "Candidates"])?;
            let id = u32_bound(&action["Candidates"], MAX_HANDLE)?;
            if candidates.remove(&id).is_none() {
                return Err(invalid("unknown candidate handle"));
            }
        }
        "snapshot-candidates" => {
            fields(action, &["Kind", "Candidates"])?;
            let id = u32_bound(&action["Candidates"], MAX_HANDLE)?;
            let view = candidates
                .get(&id)
                .ok_or_else(|| invalid("unknown candidate handle"))?;
            event = snapshot_candidates(view)?;
        }
        "apply" => {
            fields(action, &["Kind", "Sampler", "Candidates"])?;
            let sid = u32_bound(&action["Sampler"], MAX_HANDLE)?;
            let cid = u32_bound(&action["Candidates"], MAX_HANDLE)?;
            let sampler = samplers
                .get_mut(&sid)
                .ok_or_else(|| invalid("unknown sampler handle"))?;
            let view = candidates
                .get_mut(&cid)
                .ok_or_else(|| invalid("unknown candidate handle"))?;
            sampler.apply(view)?;
            if !view.validate() {
                return Err(invalid("candidate view corrupted by sampler apply"));
            }
        }
        "accept" => {
            fields(action, &["Kind", "Sampler", "Token"])?;
            let sid = u32_bound(&action["Sampler"], MAX_HANDLE)?;
            let tok = i32_val(&action["Token"])?;
            samplers
                .get_mut(&sid)
                .ok_or_else(|| invalid("unknown sampler handle"))?
                .accept(tok);
        }
        "reset" => {
            fields(action, &["Kind", "Sampler"])?;
            let sid = u32_bound(&action["Sampler"], MAX_HANDLE)?;
            samplers
                .get_mut(&sid)
                .ok_or_else(|| invalid("unknown sampler handle"))?
                .reset();
        }
        "free-sampler" => {
            fields(action, &["Kind", "Sampler"])?;
            let id = u32_bound(&action["Sampler"], MAX_HANDLE)?;
            if samplers.remove(&id).is_none() {
                return Err(invalid("unknown sampler handle"));
            }
        }
        _ => return Err(invalid("unknown sampler action Kind")),
    }
    let retained: usize = candidates.values().map(|view| view.backing.len()).sum();
    if retained > MAX_RETAINED {
        return Err(invalid("retained candidates exceed bound"));
    }
    Ok(event)
}

fn execute_trace(operation: &Value) -> Result<Value> {
    fields(operation, &["Kind", "Actions"])?;
    if text(&operation["Kind"])? != "sampler-trace" {
        return Err(invalid("unexpected sampler operation kind"));
    }
    let actions = array(&operation["Actions"])?;
    if actions.is_empty() || actions.len() > 4096 {
        return Err(invalid("sampler action count outside bound"));
    }
    let mut samplers = BTreeMap::new();
    let mut candidates = BTreeMap::new();
    let clock = Instant::now();
    let mut trace = Vec::with_capacity(actions.len());
    for action in actions {
        trace.push(execute_trace_action(
            action,
            &clock,
            &mut samplers,
            &mut candidates,
        )?);
    }
    Ok(json!({"Trace": trace}))
}

fn fields(value: &Value, names: &[&str]) -> Result<()> {
    let obj = value
        .as_object()
        .ok_or_else(|| invalid("sampler message must be object"))?;
    if obj.len() != names.len() || names.iter().any(|name| !obj.contains_key(*name)) {
        return Err(invalid("sampler fields mismatch"));
    }
    Ok(())
}

fn array(value: &Value) -> Result<&[Value]> {
    value
        .as_array()
        .map(Vec::as_slice)
        .ok_or_else(|| invalid("sampler array required"))
}

fn text(value: &Value) -> Result<&str> {
    value
        .as_str()
        .ok_or_else(|| invalid("sampler text required"))
}

fn i64_val(value: &Value) -> Result<i64> {
    value
        .as_i64()
        .ok_or_else(|| invalid("sampler value must be signed integer"))
}

fn i32_val(value: &Value) -> Result<i32> {
    Ok(i32::try_from(i64_val(value)?)?)
}

fn u32_bound(value: &Value, maximum: u32) -> Result<u32> {
    let n = value
        .as_u64()
        .ok_or_else(|| invalid("sampler value must be nonnegative integer"))?;
    let v = u32::try_from(n)?;
    if v > maximum {
        return Err(invalid("sampler unsigned value outside bound"));
    }
    Ok(v)
}

fn raw_float(value: &Value) -> Result<f32> {
    Ok(f32::from_bits(u32_bound(value, u32::MAX)?))
}

fn finite_float(value: &Value) -> Result<f32> {
    let f = raw_float(value)?;
    if !f.is_finite() {
        return Err(invalid("sampler parameter must be finite"));
    }
    Ok(f)
}

pub(super) fn handles(request: &Value) -> bool {
    request["Operations"].as_array().is_some_and(|operations| {
        !operations.is_empty()
            && operations
                .iter()
                .all(|operation| operation["Kind"] == "sampler-trace")
    })
}

pub(super) fn execute(request: &Value) -> Result<Value> {
    fields(request, &["SchemaVersion", "Operations"])?;
    if request["SchemaVersion"] != 1 {
        return Err(invalid("sampling request schema version must equal 1"));
    }
    let operations = array(&request["Operations"])?;
    if operations.is_empty() || operations.len() > 128 {
        return Err(invalid("sampling operation count outside bound"));
    }
    let results = operations
        .iter()
        .map(execute_trace)
        .collect::<Result<Vec<_>>>()?;
    Ok(json!({"SchemaVersion": 1, "Results": results}))
}

pub(super) fn identity() -> Value {
    json!({
        "Kind": "independent-rust-sampling",
        "SupportedSourceIds": [ID],
        "UpstreamCommit": super::UPSTREAM,
        "AssertionSource": {"Path": ID, "Sha256": SOURCE_SHA256},
        "Architecture": "clean-room-msvc-mt19937-partial-sort-heap-and-stateful-candidate-view-v1",
        "NativeReferenceReplay": false,
        "ProductionRuntimeCalled": false,
        "CompleteSourceGate": false,
        "NativeRetirementQualified": false
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{Engine, Target};

    #[test]
    fn independent_sampling_provider_passes_complete_test_sampling_suite_across_reference_arms()
    -> Result<()> {
        for target in [Target::RustReference, Target::Cpp] {
            let engine = Engine::new(target, None, None, None, None, None)?;
            let report = crate::sampling::run(&engine)?;
            assert_eq!(report["Id"], ID);
            assert_eq!(report["Status"], "pass", "{report}");
        }
        assert!(crate::reference_provider::supports(ID));
        let info = identity();
        assert_eq!(info["SupportedSourceIds"], json!([ID]));
        assert_eq!(info["NativeReferenceReplay"], false);
        assert_eq!(info["ProductionRuntimeCalled"], false);
        assert_eq!(info["CompleteSourceGate"], false);
        assert_eq!(info["NativeRetirementQualified"], false);
        Ok(())
    }

    #[test]
    fn independent_sampling_preserves_xtc_offset_and_history_reset() -> Result<()> {
        let trace_res = execute(&json!({
            "SchemaVersion": 1,
            "Operations": [{
                "Kind": "sampler-trace",
                "Actions": [
                    {
                        "Kind": "init-candidates",
                        "Candidates": 0,
                        "Data": [
                            {"Id": 10, "LogitBits": 5.0_f32.to_bits(), "ProbabilityBits": 0.0_f32.to_bits()},
                            {"Id": 20, "LogitBits": 4.9_f32.to_bits(), "ProbabilityBits": 0.0_f32.to_bits()},
                            {"Id": 30, "LogitBits": (-10.0_f32).to_bits(), "ProbabilityBits": 0.0_f32.to_bits()}
                        ],
                        "Selected": -1,
                        "Sorted": false
                    },
                    {
                        "Kind": "init-sampler",
                        "Sampler": 0,
                        "Type": "xtc",
                        "Parameters": {
                            "ProbabilityBits": 1.0_f32.to_bits(),
                            "ThresholdBits": 0.4_f32.to_bits(),
                            "MinKeep": 1,
                            "Seed": 1
                        }
                    },
                    {"Kind": "apply", "Sampler": 0, "Candidates": 0},
                    {"Kind": "copy-candidates", "Source": 0, "Target": 1},
                    {"Kind": "snapshot-candidates", "Candidates": 1},
                    {"Kind": "free-candidates", "Candidates": 0},
                    {"Kind": "free-candidates", "Candidates": 1},
                    {"Kind": "free-sampler", "Sampler": 0},
                    {
                        "Kind": "init-sampler",
                        "Sampler": 1,
                        "Type": "penalties",
                        "Parameters": {
                            "VocabularySize": 4,
                            "PenaltyLastN": 2,
                            "RepeatBits": 2.0_f32.to_bits(),
                            "FrequencyBits": 0.5_f32.to_bits(),
                            "PresenceBits": 0.25_f32.to_bits()
                        }
                    },
                    {"Kind": "accept", "Sampler": 1, "Token": 1},
                    {
                        "Kind": "init-candidates",
                        "Candidates": 2,
                        "Data": [{"Id": 1, "LogitBits": 4.0_f32.to_bits(), "ProbabilityBits": 0.0_f32.to_bits()}],
                        "Selected": -1,
                        "Sorted": false
                    },
                    {"Kind": "apply", "Sampler": 1, "Candidates": 2},
                    {"Kind": "snapshot-candidates", "Candidates": 2},
                    {"Kind": "reset", "Sampler": 1},
                    {
                        "Kind": "init-candidates",
                        "Candidates": 3,
                        "Data": [{"Id": 1, "LogitBits": 4.0_f32.to_bits(), "ProbabilityBits": 0.0_f32.to_bits()}],
                        "Selected": -1,
                        "Sorted": false
                    },
                    {"Kind": "apply", "Sampler": 1, "Candidates": 3},
                    {"Kind": "snapshot-candidates", "Candidates": 3},
                    {"Kind": "clock"}
                ]
            }]
        }))?;
        let trace = &trace_res["Results"][0]["Trace"];
        assert_eq!(trace[4]["Size"], 2);
        assert_eq!(trace[4]["Data"][0]["Id"], 20);
        assert_eq!(
            trace[12]["Data"][0]["LogitBits"],
            (4.0_f32 / 2.0_f32 - 0.75_f32).to_bits()
        );
        assert_eq!(trace[16]["Data"][0]["LogitBits"], 4.0_f32.to_bits());
        assert!(trace[17]["Microseconds"].as_i64().is_some());
        Ok(())
    }

    #[test]
    fn independent_sampling_rejects_invalid_requests() {
        assert!(
            execute(&json!({
                "SchemaVersion": 1,
                "Operations": [{
                    "Kind": "sampler-trace",
                    "Actions": [{
                        "Kind": "init-sampler",
                        "Sampler": 0,
                        "Type": "penalties",
                        "Parameters": {
                            "VocabularySize": 0,
                            "PenaltyLastN": 2,
                            "RepeatBits": 1.0_f32.to_bits(),
                            "FrequencyBits": 0.0_f32.to_bits(),
                            "PresenceBits": 0.0_f32.to_bits()
                        }
                    }]
                }]
            }))
            .is_err()
        );
        assert!(
            execute(&json!({
                "SchemaVersion": 1,
                "Operations": [{
                    "Kind": "sampler-trace",
                    "Actions": [{
                        "Kind": "init-candidates",
                        "Candidates": 0,
                        "Data": [],
                        "Selected": -1,
                        "Sorted": false
                    }]
                }]
            }))
            .is_err()
        );
    }
}
