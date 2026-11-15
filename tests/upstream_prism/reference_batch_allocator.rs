//! Independent bitset-partitioning and owned microbatch snapshots for batch allocation.
//!
//! This module implements the bounded `batch-allocator` state machine from
//! `llama-batch.cpp` and `batch_allocator.hpp` in safe Rust. It imports neither
//! production `bonsai_prism_runtime` code nor expected-answer fixtures; immutable
//! upstream MIT source identities are reported by `identity`.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Deserializer};
use serde_json::{Value, json};

use super::{Result, invalid};

/// Upstream test source identifier covered by this independent reference.
pub(super) const ID: &str = "tests/test-batch-alloc.cpp";

const MAX_SEQ: usize = 256;
const MAX_TOKENS: usize = 64;
const MAX_OPERATIONS: usize = 128;
const MAX_ACTIONS: usize = 64;
const MAX_EMBEDDING_WIDTH: usize = 8;
const MAX_SEQ_PER_TOKEN: usize = 8;
const MAX_MEMORY_RANGES: usize = 16;
const MAX_RESERVE_AXIS: usize = 16;
const POS_MIN: i64 = -32_768;
const POS_MAX: i64 = 32_768;

#[derive(Deserialize)]
#[serde(rename_all = "PascalCase", deny_unknown_fields)]
struct Request {
    schema_version: u32,
    operations: Vec<RawOperation>,
}

#[derive(Deserialize)]
#[serde(rename_all = "PascalCase", deny_unknown_fields)]
struct RawOperation {
    kind: String,
    positions_per_embedding: i64,
    actions: Vec<Value>,
}

#[derive(Deserialize)]
#[serde(rename_all = "PascalCase", deny_unknown_fields)]
struct MemoryRange {
    sequence: i64,
    min: i64,
    max: i64,
}

#[derive(Deserialize)]
#[serde(rename_all = "PascalCase", deny_unknown_fields)]
struct RawInit {
    op: String,
    embedding_width: i64,
    #[serde(deserialize_with = "nullable")]
    embeddings: Option<Vec<f32>>,
    #[serde(deserialize_with = "nullable")]
    tokens: Option<Vec<i64>>,
    #[serde(deserialize_with = "nullable")]
    positions: Option<Vec<i64>>,
    #[serde(deserialize_with = "nullable")]
    sequence_ids: Option<Vec<Vec<i64>>>,
    #[serde(deserialize_with = "nullable")]
    logits: Option<Vec<i64>>,
    sequence_limit: i64,
    output_all: bool,
    #[serde(deserialize_with = "nullable")]
    memory_ranges: Option<Vec<MemoryRange>>,
}

#[derive(Deserialize)]
#[serde(rename_all = "PascalCase", deny_unknown_fields)]
struct RawSplitEqual {
    op: String,
    capacity: i64,
    sequential: bool,
    keep_tail: i64,
}

#[derive(Deserialize)]
#[serde(rename_all = "PascalCase", deny_unknown_fields)]
struct RawSplitCapacity {
    op: String,
    capacity: i64,
}

#[derive(Deserialize)]
#[serde(rename_all = "PascalCase", deny_unknown_fields)]
struct RawSplitReset {
    op: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "PascalCase", deny_unknown_fields)]
struct RawReserve {
    op: String,
    sequence_tokens: i64,
    sequences: i64,
}

fn nullable<'de, T: Deserialize<'de>, D: Deserializer<'de>>(
    input: D,
) -> std::result::Result<Option<T>, D::Error> {
    Option::deserialize(input)
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
struct SeqSet([u64; 4]);

impl SeqSet {
    const fn empty() -> Self {
        Self([0; 4])
    }

    const fn full() -> Self {
        Self([u64::MAX; 4])
    }

    fn set(&mut self, id: usize) -> Result<()> {
        let word = self
            .0
            .get_mut(id / 64)
            .ok_or_else(|| invalid("sequence bitset index out of bounds"))?;
        *word |= 1_u64 << (id % 64);
        Ok(())
    }

    const fn test(self, id: usize) -> bool {
        if id < MAX_SEQ {
            ((self.0[id / 64] >> (id % 64)) & 1) != 0
        } else {
            false
        }
    }

    const fn is_empty(self) -> bool {
        self.0[0] == 0 && self.0[1] == 0 && self.0[2] == 0 && self.0[3] == 0
    }

    const fn intersect(self, other: Self) -> Self {
        Self([
            self.0[0] & other.0[0],
            self.0[1] & other.0[1],
            self.0[2] & other.0[2],
            self.0[3] & other.0[3],
        ])
    }

    const fn intersects(self, other: Self) -> bool {
        !self.intersect(other).is_empty()
    }

    const fn is_subset_of(self, superset: Self) -> bool {
        let inter = self.intersect(superset);
        inter.0[0] == self.0[0]
            && inter.0[1] == self.0[1]
            && inter.0[2] == self.0[2]
            && inter.0[3] == self.0[3]
    }
}

type MemoryRanges = BTreeMap<usize, (i32, i32)>;

enum BatchData {
    Tokens(Vec<i32>),
    Embeddings { values: Vec<f32>, width: usize },
}

struct InitSpec {
    data: BatchData,
    n_tokens: usize,
    positions: Option<Vec<i32>>,
    sequence_ids: Option<Vec<Vec<i32>>>,
    logits: Option<Vec<i8>>,
    sequence_limit: usize,
    output_all: bool,
    memory: Option<MemoryRanges>,
}

enum Action {
    Init(Box<InitSpec>),
    SplitEqual {
        capacity: usize,
        sequential: bool,
        keep_tail: usize,
    },
    SplitSimple {
        capacity: usize,
    },
    SplitSeq {
        capacity: usize,
    },
    SplitReset,
    Reserve {
        sequence_tokens: usize,
        sequences: usize,
    },
}

struct OperationSpec {
    positions_per_embedding: usize,
    actions: Vec<Action>,
}

struct Batch {
    n_tokens: usize,
    data: BatchData,
    positions: Vec<i32>,
    sequence_ids: Vec<Vec<i32>>,
    logits: Vec<i8>,
}

struct MicroBatch {
    equal_seqs: bool,
    n_tokens: usize,
    n_seq_tokens: usize,
    n_seqs: usize,
    n_pos: usize,
    tokens: Option<Vec<i32>>,
    embeddings: Option<Vec<f32>>,
    positions: Vec<i32>,
    sequence_ids: Vec<Vec<i32>>,
    unique_sequence_ids: Vec<i32>,
    sequence_indices: Vec<i32>,
    output: Vec<i8>,
}

fn bounded_i32(value: i64, minimum: i64, maximum: i64, label: &str) -> Result<i32> {
    if !(minimum..=maximum).contains(&value) {
        return Err(invalid(label));
    }
    Ok(i32::try_from(value)?)
}

fn bounded_usize(value: i64, minimum: i64, maximum: i64, label: &str) -> Result<usize> {
    if !(minimum..=maximum).contains(&value) {
        return Err(invalid(label));
    }
    Ok(usize::try_from(value)?)
}

fn parse_batch_data(raw: &mut RawInit, width: usize) -> Result<(BatchData, usize)> {
    match (raw.embeddings.take(), raw.tokens.take()) {
        (Some(values), None) if width > 0 => {
            if !values.len().is_multiple_of(width) || !values.iter().copied().all(f32::is_finite) {
                return Err(invalid(
                    "batch allocator embeddings must be finite and width-aligned",
                ));
            }
            let count = values.len() / width;
            Ok((BatchData::Embeddings { values, width }, count))
        }
        (None, Some(tokens)) if width == 0 => {
            let count = tokens.len();
            let converted = tokens
                .into_iter()
                .map(|token| {
                    bounded_i32(token, POS_MIN, POS_MAX, "batch token outside bound")
                })
                .collect::<Result<Vec<_>>>()?;
            Ok((BatchData::Tokens(converted), count))
        }
        _ => Err(invalid(
            "supply either tokens with width 0 or embeddings with positive width",
        )),
    }
}

fn parse_memory_ranges(raw: Option<Vec<MemoryRange>>) -> Result<Option<MemoryRanges>> {
    let Some(ranges) = raw else {
        return Ok(None);
    };
    if ranges.len() > MAX_MEMORY_RANGES {
        return Err(invalid("batch allocator memory range count outside bound"));
    }
    let mut map = MemoryRanges::new();
    for range in ranges {
        let id = bounded_usize(
            range.sequence,
            0,
            i64::try_from(MAX_SEQ - 1)?,
            "batch allocator memory sequence outside bound",
        )?;
        let min = bounded_i32(range.min, -1, POS_MAX, "batch memory min outside bound")?;
        let max = bounded_i32(range.max, -1, POS_MAX, "batch memory max outside bound")?;
        if map.insert(id, (min, max)).is_some() {
            return Err(invalid("duplicate batch allocator memory sequence range"));
        }
    }
    Ok(Some(map))
}

fn validate_init(value: &Value, n_pos: usize) -> Result<Action> {
    let mut raw: RawInit = serde_json::from_value(value.clone())?;
    if raw.op != "init" {
        return Err(invalid("batch allocator init Op mismatch"));
    }
    let width = bounded_usize(
        raw.embedding_width,
        0,
        i64::try_from(MAX_EMBEDDING_WIDTH)?,
        "batch allocator EmbeddingWidth outside bound",
    )?;
    let sequence_limit = bounded_usize(
        raw.sequence_limit,
        1,
        i64::try_from(MAX_SEQ + 1)?,
        "batch allocator SequenceLimit outside bound",
    )?;
    let (data, count) = parse_batch_data(&mut raw, width)?;
    if !(1..=MAX_TOKENS).contains(&count) {
        return Err(invalid("batch allocator token count outside bound"));
    }
    let pos_factor = match data {
        BatchData::Embeddings { .. } => n_pos,
        BatchData::Tokens(_) => 1,
    };
    let positions = raw
        .positions
        .map(|items| {
            if items.len() != count * pos_factor {
                return Err(invalid("batch allocator Positions extent mismatch"));
            }
            items
                .into_iter()
                .map(|pos| bounded_i32(pos, POS_MIN, POS_MAX, "batch position outside bound"))
                .collect::<Result<Vec<_>>>()
        })
        .transpose()?;
    let sequence_ids = parse_sequence_ids(raw.sequence_ids, count)?;
    let logits = parse_logits(raw.logits, count)?;
    let memory = parse_memory_ranges(raw.memory_ranges)?;
    Ok(Action::Init(Box::new(InitSpec {
        data,
        n_tokens: count,
        positions,
        sequence_ids,
        logits,
        sequence_limit,
        output_all: raw.output_all,
        memory,
    })))
}

fn parse_sequence_ids(raw: Option<Vec<Vec<i64>>>, count: usize) -> Result<Option<Vec<Vec<i32>>>> {
    let Some(lists) = raw else {
        return Ok(None);
    };
    if lists.len() != count {
        return Err(invalid("batch allocator SequenceIds extent mismatch"));
    }
    let max_seq = i64::try_from(MAX_SEQ)?;
    lists
        .into_iter()
        .map(|ids| {
            if !(1..=MAX_SEQ_PER_TOKEN).contains(&ids.len()) {
                return Err(invalid("batch allocator sequence list outside bound"));
            }
            ids.into_iter()
                .map(|id| bounded_i32(id, -1, max_seq, "batch sequence ID outside bound"))
                .collect::<Result<Vec<_>>>()
        })
        .collect::<Result<Vec<_>>>()
        .map(Some)
}

fn parse_logits(raw: Option<Vec<i64>>, count: usize) -> Result<Option<Vec<i8>>> {
    let Some(items) = raw else {
        return Ok(None);
    };
    if items.len() != count {
        return Err(invalid("batch allocator Logits extent mismatch"));
    }
    items
        .into_iter()
        .map(|flag| {
            if !(0..=1).contains(&flag) {
                return Err(invalid("batch allocator Logits flag outside bound"));
            }
            Ok(i8::try_from(flag)?)
        })
        .collect::<Result<Vec<_>>>()
        .map(Some)
}

fn parse_action(value: &Value, n_pos: usize) -> Result<Action> {
    let op = value
        .as_object()
        .and_then(|object| object.get("Op"))
        .and_then(Value::as_str)
        .ok_or_else(|| invalid("batch allocator action Op required"))?;
    let max_tokens = i64::try_from(MAX_TOKENS)?;
    match op {
        "init" => validate_init(value, n_pos),
        "split-equal" => {
            let raw: RawSplitEqual = serde_json::from_value(value.clone())?;
            if raw.op != "split-equal" {
                return Err(invalid("batch allocator split-equal Op mismatch"));
            }
            let capacity = bounded_usize(raw.capacity, 1, max_tokens, "Capacity outside bound")?;
            let keep_tail = bounded_usize(raw.keep_tail, 0, max_tokens, "KeepTail outside bound")?;
            Ok(Action::SplitEqual {
                capacity,
                sequential: raw.sequential,
                keep_tail,
            })
        }
        "split-simple" | "split-seq" => {
            let raw: RawSplitCapacity = serde_json::from_value(value.clone())?;
            let capacity = bounded_usize(raw.capacity, 1, max_tokens, "Capacity outside bound")?;
            Ok(if raw.op == "split-simple" {
                Action::SplitSimple { capacity }
            } else {
                Action::SplitSeq { capacity }
            })
        }
        "split-reset" => {
            let raw: RawSplitReset = serde_json::from_value(value.clone())?;
            if raw.op != "split-reset" {
                return Err(invalid("batch allocator split-reset Op mismatch"));
            }
            Ok(Action::SplitReset)
        }
        "reserve" => {
            let raw: RawReserve = serde_json::from_value(value.clone())?;
            if raw.op != "reserve" {
                return Err(invalid("batch allocator reserve Op mismatch"));
            }
            let max_axis = i64::try_from(MAX_RESERVE_AXIS)?;
            let sequence_tokens = bounded_usize(
                raw.sequence_tokens,
                1,
                max_axis,
                "SequenceTokens outside bound",
            )?;
            let sequences =
                bounded_usize(raw.sequences, 1, max_axis, "Sequences outside bound")?;
            if sequence_tokens * sequences > MAX_TOKENS {
                return Err(invalid("reserved batch exceeds token count bound"));
            }
            Ok(Action::Reserve {
                sequence_tokens,
                sequences,
            })
        }
        _ => Err(invalid("unknown batch allocator action Op")),
    }
}

fn validate_operation(raw: &RawOperation) -> Result<OperationSpec> {
    if raw.kind != "batch-allocator" {
        return Err(invalid("unknown batch allocator Kind"));
    }
    if raw.positions_per_embedding != 1 && raw.positions_per_embedding != 4 {
        return Err(invalid("PositionsPerEmbedding must be 1 or 4"));
    }
    let n_pos = usize::try_from(raw.positions_per_embedding)?;
    if !(1..=MAX_ACTIONS).contains(&raw.actions.len()) {
        return Err(invalid("batch allocator action count outside bound"));
    }
    let actions = raw
        .actions
        .iter()
        .map(|action| parse_action(action, n_pos))
        .collect::<Result<Vec<_>>>()?;
    Ok(OperationSpec {
        positions_per_embedding: n_pos,
        actions,
    })
}

struct BatchAllocator {
    n_pos_per_embd: usize,
    n_seq_max: usize,
    n_outputs: usize,
    has_cpl: bool,
    batch: Option<Batch>,
    seq_pos: Vec<BTreeSet<i32>>,
    seq_cpl: Vec<Vec<bool>>,
    seq_set: Vec<SeqSet>,
    seq_set_map: BTreeMap<SeqSet, Vec<usize>>,
    out_ids: Vec<i32>,
    n_used: usize,
    used: Vec<bool>,
}

impl BatchAllocator {
    fn new(n_pos_per_embd: usize) -> Self {
        Self {
            n_pos_per_embd,
            n_seq_max: 0,
            n_outputs: 0,
            has_cpl: false,
            batch: None,
            seq_pos: vec![BTreeSet::new(); MAX_SEQ],
            seq_cpl: vec![vec![false; MAX_SEQ]; MAX_SEQ],
            seq_set: Vec::new(),
            seq_set_map: BTreeMap::new(),
            out_ids: Vec::new(),
            n_used: 0,
            used: Vec::new(),
        }
    }

    fn clear(&mut self) {
        self.n_outputs = 0;
        self.n_seq_max = 0;
        self.has_cpl = false;
        self.batch = None;
        for set in &mut self.seq_pos {
            set.clear();
        }
        for row in &mut self.seq_cpl {
            row.fill(false);
        }
        self.seq_set.clear();
        self.seq_set_map.clear();
    }

    fn split_reset(&mut self) {
        let n_tokens = self.batch.as_ref().map_or(0, |batch| batch.n_tokens);
        self.out_ids.clear();
        self.n_used = 0;
        self.used.clear();
        self.used.resize(n_tokens, false);
    }

    fn seq_pos_min(&self, seq_id: usize) -> i32 {
        self.seq_pos
            .get(seq_id)
            .and_then(|set| set.first().copied())
            .unwrap_or(-1)
    }

    fn seq_pos_max(&self, seq_id: usize) -> i32 {
        self.seq_pos
            .get(seq_id)
            .and_then(|set| set.last().copied())
            .unwrap_or(-1)
    }

    fn init(&mut self, spec: InitSpec) -> Result<bool> {
        self.clear();
        if !validate_init_inputs(&spec) {
            return Ok(false);
        }
        let n_tokens = spec.n_tokens;
        let has_embd = matches!(spec.data, BatchData::Embeddings { .. });
        let sequence_ids = spec
            .sequence_ids
            .unwrap_or_else(|| vec![vec![0]; n_tokens]);
        let positions = spec.positions.map_or_else(
            || {
                if has_embd && self.n_pos_per_embd > 1 {
                    return Err(invalid(
                        "embedding batch with PositionsPerEmbedding > 1 requires explicit Positions",
                    ));
                }
                autofill_positions(&sequence_ids, spec.memory.as_ref(), spec.sequence_limit)
            },
            Ok,
        )?;
        let logits = autofill_logits(spec.logits, n_tokens, spec.output_all);
        let batch = Batch {
            n_tokens,
            data: spec.data,
            positions,
            sequence_ids,
            logits,
        };
        self.n_seq_max = spec.sequence_limit;
        self.populate_sequences(&batch)?;
        if !self.check_consistency(&batch, spec.memory.as_ref())? {
            return Ok(false);
        }
        self.batch = Some(batch);
        self.split_reset();
        Ok(true)
    }

    fn populate_sequences(&mut self, batch: &Batch) -> Result<()> {
        self.n_outputs = batch.logits.iter().filter(|&&flag| flag != 0).count();
        self.has_cpl = false;
        for (i, ids) in batch.sequence_ids.iter().enumerate() {
            let pos = *batch
                .positions
                .get(i)
                .ok_or_else(|| invalid("batch position index out of bounds"))?;
            let s0 = usize::try_from(ids[0])?;
            let mut cur = SeqSet::empty();
            for (s_idx, &id) in ids.iter().enumerate() {
                let s1 = usize::try_from(id)?;
                self.seq_pos
                    .get_mut(s1)
                    .ok_or_else(|| invalid("sequence ID out of range"))?
                    .insert(pos);
                cur.set(s1)?;
                if s_idx > 0 {
                    self.seq_cpl[s1][s0] = true;
                    self.has_cpl = true;
                }
            }
            self.seq_set.push(cur);
            self.seq_set_map.entry(cur).or_default().push(i);
        }
        Ok(())
    }

    fn check_consistency(&self, batch: &Batch, memory: Option<&MemoryRanges>) -> Result<bool> {
        let has_tokens = matches!(batch.data, BatchData::Tokens(_));
        if !self.check_memory_ranges(has_tokens, memory)? {
            return Ok(false);
        }
        if let Some(mem) = memory
            && !self.check_coupled_sequences(mem)
        {
            return Ok(false);
        }
        self.check_sequence_order(batch)
    }

    fn check_memory_ranges(
        &self,
        has_tokens: bool,
        memory: Option<&MemoryRanges>,
    ) -> Result<bool> {
        for s in 0..self.n_seq_max {
            if self.seq_pos[s].is_empty() {
                continue;
            }
            let p0 = memory.and_then(|mem| mem.get(&s)).map_or(-1, |range| range.1);
            let min_pos = self.seq_pos_min(s);
            if self.n_pos_per_embd > 1 {
                let bad = if has_tokens {
                    p0 >= 0 && p0 >= min_pos
                } else {
                    p0 >= 0 && p0 > min_pos
                };
                if bad {
                    return Ok(false);
                }
            } else {
                if p0 >= 0 && min_pos != p0 + 1 {
                    return Ok(false);
                }
                let span = i64::from(self.seq_pos_max(s)) - i64::from(min_pos) + 1;
                if span > i64::try_from(self.seq_pos[s].len())? {
                    return Ok(false);
                }
            }
        }
        Ok(true)
    }

    fn check_coupled_sequences(&self, mem: &MemoryRanges) -> bool {
        for s0 in 0..self.n_seq_max {
            for s1 in 0..self.n_seq_max {
                if self.seq_cpl[s0][s1] {
                    let r0 = mem.get(&s0).copied().unwrap_or((-1, -1));
                    let r1 = mem.get(&s1).copied().unwrap_or((-1, -1));
                    if r0 != r1 {
                        return false;
                    }
                }
            }
        }
        true
    }

    fn check_sequence_order(&self, batch: &Batch) -> Result<bool> {
        let mut cur_seq_set = vec![SeqSet::full(); self.n_seq_max];
        let mut cur_seq_pos = vec![-1_i32; self.n_seq_max];
        for (i, ids) in batch.sequence_ids.iter().enumerate() {
            let pos = *batch
                .positions
                .get(i)
                .ok_or_else(|| invalid("batch position index out of bounds"))?;
            for &id in ids {
                let seq = usize::try_from(id)?;
                cur_seq_set[seq] = cur_seq_set[seq].intersect(self.seq_set[i]);
                if cur_seq_set[seq].is_empty() || pos < cur_seq_pos[seq] {
                    return Ok(false);
                }
                cur_seq_pos[seq] = pos;
            }
        }
        Ok(true)
    }

    fn split_simple(&mut self, n_ubatch: usize) -> Result<Option<MicroBatch>> {
        let Some(mut cur_idx) = self.used.iter().position(|&used| !used) else {
            return Ok(None);
        };
        let total = self.used.len();
        let mut idxs = Vec::new();
        loop {
            idxs.push(cur_idx);
            self.used[cur_idx] = true;
            self.n_used += 1;
            cur_idx += 1;
            if cur_idx >= total || idxs.len() >= n_ubatch {
                break;
            }
        }
        self.ubatch_add(&idxs, idxs.len(), false).map(Some)
    }

    fn split_seq(&mut self, n_ubatch: usize) -> Result<Option<MicroBatch>> {
        let Some(mut cur_idx) = self.used.iter().position(|&used| !used) else {
            return Ok(None);
        };
        let total = self.used.len();
        let mut cur_seq_set = self.seq_set[cur_idx];
        let mut idxs = Vec::new();
        loop {
            idxs.push(cur_idx);
            self.used[cur_idx] = true;
            self.n_used += 1;
            if idxs.len() >= n_ubatch {
                break;
            }
            cur_idx += 1;
            while cur_idx < total
                && (self.used[cur_idx] || !self.seq_set[cur_idx].is_subset_of(cur_seq_set))
            {
                cur_idx += 1;
            }
            if cur_idx == total {
                break;
            }
            cur_seq_set = self.seq_set[cur_idx];
        }
        self.ubatch_add(&idxs, 1, true).map(Some)
    }

    fn split_equal(
        &mut self,
        n_ubatch: usize,
        sequential: bool,
        n_keep_tail: usize,
    ) -> Result<Option<MicroBatch>> {
        if n_keep_tail > 0 && n_ubatch <= n_keep_tail {
            return Err(invalid("split-equal with KeepTail requires Capacity > KeepTail"));
        }
        if sequential && self.has_cpl {
            return Ok(None);
        }
        let cur_seq_sets = self.select_equal_seq_sets(n_ubatch, sequential)?;
        if cur_seq_sets.is_empty() {
            return Ok(None);
        }
        let mut cur_idx = self.first_unused_indices(&cur_seq_sets)?;
        let mut idxs_per_seq = self.expand_equal_seqs(&cur_seq_sets, &mut cur_idx, n_ubatch)?;
        let n_seqs = if n_keep_tail > 0 {
            self.apply_keep_tail(
                &cur_seq_sets,
                &mut cur_idx,
                &mut idxs_per_seq,
                n_ubatch,
                n_keep_tail,
            )?
        } else {
            cur_seq_sets.len()
        };
        let mut idxs = Vec::new();
        for seq_idxs in idxs_per_seq.into_iter().take(n_seqs) {
            idxs.extend(seq_idxs);
        }
        self.ubatch_add(&idxs, n_seqs, true).map(Some)
    }

    fn select_equal_seq_sets(&self, n_ubatch: usize, sequential: bool) -> Result<Vec<SeqSet>> {
        let batch = self
            .batch
            .as_ref()
            .ok_or_else(|| invalid("batch allocator has no initialized input"))?;
        let mut cur_seq_sets: Vec<SeqSet> = Vec::new();
        let mut last_seq_id = -1_i32;
        for (i, ids) in batch.sequence_ids.iter().enumerate() {
            if self.used[i] {
                continue;
            }
            let set = self.seq_set[i];
            let overlaps = cur_seq_sets.iter().any(|existing| existing.intersects(set));
            let seq_ok = !sequential || cur_seq_sets.is_empty() || ids[0] == last_seq_id + 1;
            if !overlaps && seq_ok {
                cur_seq_sets.push(set);
                last_seq_id = ids[0];
                if cur_seq_sets.len() > n_ubatch {
                    break;
                }
            }
        }
        Ok(cur_seq_sets)
    }

    fn first_unused_indices(&self, sets: &[SeqSet]) -> Result<Vec<usize>> {
        let mut cur_idx = Vec::with_capacity(sets.len());
        for set in sets {
            let list = self
                .seq_set_map
                .get(set)
                .ok_or_else(|| invalid("missing sequence set map entry"))?;
            let pos = list
                .iter()
                .position(|&batch_idx| !self.used[batch_idx])
                .ok_or_else(|| invalid("expected unused token in sequence set"))?;
            cur_idx.push(pos);
        }
        Ok(cur_idx)
    }

    fn expand_equal_seqs(
        &mut self,
        sets: &[SeqSet],
        cur_idx: &mut [usize],
        n_ubatch: usize,
    ) -> Result<Vec<Vec<usize>>> {
        let n_seqs = sets.len();
        let mut idxs_per_seq = vec![Vec::new(); n_seqs];
        loop {
            let can_expand = sets.iter().enumerate().all(|(s, set)| {
                self.seq_set_map
                    .get(set)
                    .is_some_and(|list| cur_idx[s] < list.len())
            });
            if !can_expand {
                break;
            }
            for (s, set) in sets.iter().enumerate() {
                let list = self
                    .seq_set_map
                    .get(set)
                    .ok_or_else(|| invalid("missing sequence set map entry"))?;
                let batch_idx = list[cur_idx[s]];
                idxs_per_seq[s].push(batch_idx);
                self.used[batch_idx] = true;
                self.n_used += 1;
                cur_idx[s] += 1;
            }
            if (idxs_per_seq[0].len() + 1) * n_seqs > n_ubatch {
                break;
            }
        }
        Ok(idxs_per_seq)
    }

    fn apply_keep_tail(
        &mut self,
        sets: &[SeqSet],
        cur_idx: &mut [usize],
        idxs_per_seq: &mut [Vec<usize>],
        n_ubatch: usize,
        n_keep_tail: usize,
    ) -> Result<usize> {
        let remaining_for = |s: usize, idx: usize| -> Result<usize> {
            let total = self
                .seq_set_map
                .get(&sets[s])
                .ok_or_else(|| invalid("missing sequence set map entry"))?
                .len();
            Ok(total - idx)
        };
        let mut n_keep = 0;
        while n_keep < sets.len() {
            let remaining = remaining_for(n_keep, cur_idx[n_keep])?;
            if remaining != 0 && remaining < n_keep_tail {
                break;
            }
            n_keep += 1;
        }
        if n_keep == 0 {
            let list = self
                .seq_set_map
                .get(&sets[0])
                .ok_or_else(|| invalid("missing sequence set map entry"))?;
            let rem = list.len() - cur_idx[0];
            if idxs_per_seq[0].len() + rem <= n_ubatch {
                while cur_idx[0] < list.len() {
                    let batch_idx = list[cur_idx[0]];
                    idxs_per_seq[0].push(batch_idx);
                    self.used[batch_idx] = true;
                    self.n_used += 1;
                    cur_idx[0] += 1;
                }
            } else {
                while list.len() - cur_idx[0] < n_keep_tail {
                    let batch_idx = idxs_per_seq[0]
                        .pop()
                        .ok_or_else(|| invalid("keep_tail truncated past sequence start"))?;
                    self.used[batch_idx] = false;
                    self.n_used -= 1;
                    cur_idx[0] -= 1;
                }
            }
            n_keep = 1;
        }
        for deferred in idxs_per_seq.iter().skip(n_keep) {
            for &batch_idx in deferred {
                self.used[batch_idx] = false;
                self.n_used -= 1;
            }
        }
        Ok(n_keep)
    }

    fn ubatch_add(
        &mut self,
        idxs: &[usize],
        n_seqs: usize,
        equal_seqs: bool,
    ) -> Result<MicroBatch> {
        let n_tokens = idxs.len();
        if n_seqs == 0 || !n_tokens.is_multiple_of(n_seqs) {
            return Err(invalid("ubatch token count is not divisible by n_seqs"));
        }
        let batch = self
            .batch
            .as_ref()
            .ok_or_else(|| invalid("batch allocator has no initialized input"))?;
        let mut tokens = matches!(batch.data, BatchData::Tokens(_))
            .then(|| Vec::with_capacity(n_tokens));
        let mut embeddings = match batch.data {
            BatchData::Embeddings { width, .. } => Some(Vec::with_capacity(n_tokens * width)),
            BatchData::Tokens(_) => None,
        };
        let mut sequence_ids = Vec::with_capacity(n_tokens);
        let mut output = Vec::with_capacity(n_tokens);
        let mut seq_set_unq = SeqSet::empty();
        for &idx in idxs {
            match &batch.data {
                BatchData::Tokens(src) => {
                    if let Some(dst) = &mut tokens {
                        dst.push(src[idx]);
                    }
                }
                BatchData::Embeddings { values, width } => {
                    if let Some(dst) = &mut embeddings {
                        let start = idx * width;
                        dst.extend_from_slice(&values[start..start + width]);
                    }
                }
            }
            sequence_ids.push(batch.sequence_ids[idx].clone());
            for &id in &batch.sequence_ids[idx] {
                seq_set_unq.set(usize::try_from(id)?)?;
            }
            let flag = batch.logits[idx];
            output.push(flag);
            if flag != 0 {
                self.out_ids.push(i32::try_from(idx)?);
            }
        }
        let has_tokens = tokens.is_some();
        let positions = collect_ubatch_positions(batch, self.n_pos_per_embd, idxs, has_tokens)?;
        let (unique_sequence_ids, sequence_indices) =
            collect_unique_sequences(seq_set_unq, self.n_seq_max)?;
        Ok(MicroBatch {
            equal_seqs,
            n_tokens,
            n_seq_tokens: n_tokens / n_seqs,
            n_seqs,
            n_pos: self.n_pos_per_embd,
            tokens,
            embeddings,
            positions,
            sequence_ids,
            unique_sequence_ids,
            sequence_indices,
            output,
        })
    }

    fn ubatch_reserve(&mut self, n_seq_tokens: usize, n_seqs: usize) -> Result<MicroBatch> {
        let n_tokens = n_seq_tokens * n_seqs;
        self.clear();
        self.split_reset();
        let mut sequence_indices = vec![-1_i32; MAX_SEQ];
        let mut unique_sequence_ids = Vec::with_capacity(n_seqs);
        for s in 0..n_seqs {
            let id = i32::try_from(s)?;
            sequence_indices[s] = id;
            unique_sequence_ids.push(id);
        }
        Ok(MicroBatch {
            equal_seqs: true,
            n_tokens,
            n_seq_tokens,
            n_seqs,
            n_pos: self.n_pos_per_embd,
            tokens: Some(vec![0; n_tokens]),
            embeddings: None,
            positions: vec![0; n_tokens * self.n_pos_per_embd],
            sequence_ids: vec![Vec::new(); n_tokens],
            unique_sequence_ids,
            sequence_indices,
            output: vec![0; n_tokens],
        })
    }
}

fn validate_init_inputs(spec: &InitSpec) -> bool {
    if spec.sequence_limit > MAX_SEQ {
        return false;
    }
    // Empty vocab has vocab.n_tokens() == 0, rejecting any supplied token id.
    if matches!(spec.data, BatchData::Tokens(_)) {
        return false;
    }
    if let Some(seq_ids) = &spec.sequence_ids {
        let Ok(limit) = i32::try_from(spec.sequence_limit) else {
            return false;
        };
        for ids in seq_ids {
            if ids.iter().any(|&id| !(0..limit).contains(&id)) {
                return false;
            }
        }
    }
    true
}

fn collect_ubatch_positions(
    batch: &Batch,
    n_pos_per_embd: usize,
    idxs: &[usize],
    has_tokens: bool,
) -> Result<Vec<i32>> {
    let mut positions = Vec::with_capacity(idxs.len() * n_pos_per_embd);
    for j in 0..n_pos_per_embd {
        let src_off = if has_tokens { 0 } else { j * batch.n_tokens };
        for &idx in idxs {
            let value = batch
                .positions
                .get(src_off + idx)
                .copied()
                .ok_or_else(|| invalid("batch position index out of bounds"))?;
            positions.push(value);
        }
    }
    Ok(positions)
}

fn autofill_positions(
    sequence_ids: &[Vec<i32>],
    memory: Option<&MemoryRanges>,
    n_seq_max: usize,
) -> Result<Vec<i32>> {
    let mut p0 = vec![0_i32; n_seq_max];
    if let Some(mem) = memory {
        for (seq, slot) in p0.iter_mut().enumerate() {
            *slot = mem.get(&seq).map_or(-1, |range| range.1) + 1;
        }
    }
    let mut positions = Vec::with_capacity(sequence_ids.len());
    for ids in sequence_ids {
        let first = usize::try_from(ids[0])?;
        let pos = *p0
            .get(first)
            .ok_or_else(|| invalid("sequence ID out of range"))?;
        positions.push(pos);
        for &id in ids {
            let idx = usize::try_from(id)?;
            let slot = p0
                .get_mut(idx)
                .ok_or_else(|| invalid("sequence ID out of range"))?;
            *slot = pos + 1;
        }
    }
    Ok(positions)
}

fn autofill_logits(raw: Option<Vec<i8>>, n_tokens: usize, output_all: bool) -> Vec<i8> {
    raw.map_or_else(
        || {
            let mut flags = vec![i8::from(output_all); n_tokens];
            if !output_all && let Some(last) = flags.last_mut() {
                *last = 1;
            }
            flags
        },
        |mut flags| {
            if output_all && flags.iter().any(|&flag| flag == 0) {
                flags.fill(1);
            }
            flags
        },
    )
}

fn collect_unique_sequences(
    seq_set_unq: SeqSet,
    n_seq_max: usize,
) -> Result<(Vec<i32>, Vec<i32>)> {
    let mut seq_id_unq = Vec::new();
    let mut seq_idx = vec![-1_i32; MAX_SEQ];
    for s in 0..n_seq_max {
        if seq_set_unq.test(s) {
            seq_idx[s] = i32::try_from(seq_id_unq.len())?;
            seq_id_unq.push(i32::try_from(s)?);
        }
    }
    Ok((seq_id_unq, seq_idx))
}

fn batch_json(batch: &Batch) -> Value {
    let (tokens, embeddings) = match &batch.data {
        BatchData::Tokens(tokens) => (json!(tokens), Value::Null),
        BatchData::Embeddings { values, .. } => (Value::Null, json!(values)),
    };
    json!({
        "Embeddings": embeddings,
        "Tokens": tokens,
        "Positions": batch.positions,
        "SequenceIds": batch.sequence_ids,
        "Logits": batch.logits,
    })
}

fn state_json(allocator: &BatchAllocator, limit: usize) -> Result<Value> {
    let batch = allocator
        .batch
        .as_ref()
        .ok_or_else(|| invalid("batch allocator has no initialized input"))?;
    let positions = (0..limit)
        .map(|id| {
            json!({
                "Min": allocator.seq_pos_min(id),
                "Max": allocator.seq_pos_max(id),
            })
        })
        .collect::<Vec<_>>();
    Ok(json!({
        "NTokens": batch.n_tokens,
        "NOutputs": allocator.n_outputs,
        "NUsed": allocator.n_used,
        "Batch": batch_json(batch),
        "OutIds": allocator.out_ids,
        "SequencePositions": positions,
    }))
}

fn ubatch_json(ubatch: Option<&MicroBatch>) -> Value {
    let Some(ub) = ubatch else {
        return json!({
            "NTokens": 0,
            "NSequenceTokens": 0,
            "NSequences": 0,
            "NUniqueSequences": 0,
            "NPositions": 0,
            "EqualSequences": false,
            "IsPosition2D": false,
            "HasTokens": false,
            "HasEmbeddings": false,
            "Tokens": Value::Null,
            "Embeddings": Value::Null,
            "Positions": Value::Null,
            "SequenceIds": Value::Null,
            "UniqueSequenceIds": Value::Null,
            "SequenceIndices": Value::Null,
            "Output": Value::Null,
        });
    };
    json!({
        "NTokens": ub.n_tokens,
        "NSequenceTokens": ub.n_seq_tokens,
        "NSequences": ub.n_seqs,
        "NUniqueSequences": ub.unique_sequence_ids.len(),
        "NPositions": ub.n_pos,
        "EqualSequences": ub.equal_seqs,
        "IsPosition2D": ub.n_pos >= 3,
        "HasTokens": ub.tokens.is_some(),
        "HasEmbeddings": ub.embeddings.is_some(),
        "Tokens": ub.tokens,
        "Embeddings": ub.embeddings,
        "Positions": ub.positions,
        "SequenceIds": ub.sequence_ids,
        "UniqueSequenceIds": ub.unique_sequence_ids,
        "SequenceIndices": ub.sequence_indices,
        "Output": ub.output,
    })
}

fn require_initialized(initialized: bool) -> Result<()> {
    if initialized {
        Ok(())
    } else {
        Err(invalid(
            "cannot split or reset an uninitialized or rejected batch",
        ))
    }
}

fn run_action(
    allocator: &mut BatchAllocator,
    limit: &mut usize,
    initialized: &mut bool,
    action: Action,
) -> Result<Value> {
    match action {
        Action::Init(spec) => {
            *limit = spec.sequence_limit;
            *initialized = allocator.init(*spec)?;
            let state = if *initialized {
                state_json(allocator, *limit)?
            } else {
                Value::Null
            };
            Ok(json!({"Initialized": *initialized, "State": state}))
        }
        Action::Reserve {
            sequence_tokens,
            sequences,
        } => {
            let ubatch = allocator.ubatch_reserve(sequence_tokens, sequences)?;
            *initialized = false;
            Ok(json!({"Ubatch": ubatch_json(Some(&ubatch)), "State": Value::Null}))
        }
        Action::SplitReset => {
            require_initialized(*initialized)?;
            allocator.split_reset();
            Ok(json!({"State": state_json(allocator, *limit)?}))
        }
        Action::SplitSimple { capacity } => {
            require_initialized(*initialized)?;
            let ubatch = allocator.split_simple(capacity)?;
            Ok(json!({
                "Ubatch": ubatch_json(ubatch.as_ref()),
                "State": state_json(allocator, *limit)?,
            }))
        }
        Action::SplitSeq { capacity } => {
            require_initialized(*initialized)?;
            let ubatch = allocator.split_seq(capacity)?;
            Ok(json!({
                "Ubatch": ubatch_json(ubatch.as_ref()),
                "State": state_json(allocator, *limit)?,
            }))
        }
        Action::SplitEqual {
            capacity,
            sequential,
            keep_tail,
        } => {
            require_initialized(*initialized)?;
            let ubatch = allocator.split_equal(capacity, sequential, keep_tail)?;
            Ok(json!({
                "Ubatch": ubatch_json(ubatch.as_ref()),
                "State": state_json(allocator, *limit)?,
            }))
        }
    }
}

fn run_operation(operation: OperationSpec) -> Result<Value> {
    let mut allocator = BatchAllocator::new(operation.positions_per_embedding);
    let mut limit = 0;
    let mut initialized = false;
    let mut observations = Vec::with_capacity(operation.actions.len());
    for action in operation.actions {
        observations.push(run_action(
            &mut allocator,
            &mut limit,
            &mut initialized,
            action,
        )?);
    }
    Ok(json!({"Actions": observations}))
}

/// Returns `true` when every operation in `value` selects `"batch-allocator"`.
pub(super) fn handles(value: &Value) -> bool {
    value["Operations"].as_array().is_some_and(|operations| {
        !operations.is_empty()
            && operations
                .iter()
                .all(|operation| operation["Kind"] == "batch-allocator")
    })
}

/// Executes a bounded `"batch-allocator"` protocol request against the independent reference.
///
/// # Errors
///
/// Returns an error when the request framing, bounds, scalar types, or action
/// sequence violates the `batch-allocator` protocol contract.
pub(super) fn execute(value: &Value) -> Result<Value> {
    let Some(operations) = value["Operations"].as_array() else {
        return Err(invalid("batch allocator Operations must be an array"));
    };
    if !(1..=MAX_OPERATIONS).contains(&operations.len()) {
        return Err(invalid("batch allocator operation count outside bound"));
    }
    for operation in operations {
        if !operation["Actions"]
            .as_array()
            .is_some_and(|actions| (1..=MAX_ACTIONS).contains(&actions.len()))
        {
            return Err(invalid("batch allocator action count outside bound"));
        }
    }
    let request: Request = serde_json::from_value(value.clone())?;
    if request.schema_version != 1 {
        return Err(invalid("unsupported batch allocator schema version"));
    }
    let validated = request
        .operations
        .iter()
        .map(validate_operation)
        .collect::<Result<Vec<_>>>()?;
    let results = validated
        .into_iter()
        .map(run_operation)
        .collect::<Result<Vec<_>>>()?;
    Ok(json!({"SchemaVersion": 1, "Results": results}))
}

/// Returns provenance and non-native execution identity metadata for `ID`.
pub(super) fn identity() -> Value {
    json!({
        "Kind": "independent-rust-batch-allocator",
        "SupportedSourceIds": [ID],
        "UpstreamCommit": super::UPSTREAM,
        "AssertionSource": {
            "Path": ID,
            "Bytes": 23_212,
            "Sha256": "341b3c4291cb46e412d6e750c2ab2e19131812d08ee3b236938c587e99341d79",
        },
        "ProductionSources": [
            {
                "Path": "src/llama-batch.cpp",
                "Bytes": 31_110,
                "Sha256": "484607d92fb08d510727e0c4b01139e27a40137ef03e74e1dbea4aac3965e328",
            },
            {
                "Path": "src/llama-batch.h",
                "Bytes": 6_580,
                "Sha256": "7000dfd8750e06c544642dd7b84c882191702ad46ced1a4d6ad109462d1a1013",
            },
        ],
        "Architecture": "bitset-sequence-partition-and-owned-ubatch-snapshot-v1",
        "OriginalVariants": 26,
        "OriginalAssertions": 198,
        "NativeReferenceReplay": false,
        "ProductionRuntimeCalled": false,
        "CompleteSourceGate": false,
        "NativeRetirementQualified": false,
    })
}

#[cfg(test)]
mod tests {
    use super::{ID, Result, execute, handles, identity};
    use serde_json::{Value, json};

    struct BatchFixture {
        embeddings: Vec<f32>,
        positions: Vec<i32>,
        sequence_ids: Vec<Vec<i32>>,
        logits: Vec<i32>,
    }

    impl BatchFixture {
        const fn new() -> Self {
            Self {
                embeddings: Vec::new(),
                positions: Vec::new(),
                sequence_ids: Vec::new(),
                logits: Vec::new(),
            }
        }

        fn add(&mut self, pos: i32, ids: &[i32], output: bool) -> Result<()> {
            let idx = f32::from(u16::try_from(self.sequence_ids.len())?);
            self.embeddings.push(idx * 100.0);
            self.embeddings.push(idx.mul_add(100.0, 1.0));
            self.positions.push(pos);
            self.sequence_ids.push(ids.to_vec());
            self.logits.push(i32::from(output));
            Ok(())
        }

        fn add_seq(&mut self, count: i32, seq_id: i32) -> Result<()> {
            for pos in 0..count {
                self.add(pos, &[seq_id], pos == count - 1)?;
            }
            Ok(())
        }

        fn init_action(&self, limit: i32, memory: Option<&[(i32, i32, i32)]>) -> Value {
            let ranges = memory.map(|entries| {
                entries
                    .iter()
                    .map(|&(sequence, min, max)| {
                        json!({"Sequence": sequence, "Min": min, "Max": max})
                    })
                    .collect::<Vec<_>>()
            });
            json!({
                "Op": "init",
                "EmbeddingWidth": 2,
                "Embeddings": self.embeddings,
                "Tokens": Value::Null,
                "Positions": self.positions,
                "SequenceIds": self.sequence_ids,
                "Logits": self.logits,
                "SequenceLimit": limit,
                "OutputAll": false,
                "MemoryRanges": ranges,
            })
        }
    }

    fn request(n_pos: u32, actions: &[Value]) -> Value {
        json!({
            "SchemaVersion": 1,
            "Operations": [{
                "Kind": "batch-allocator",
                "PositionsPerEmbedding": n_pos,
                "Actions": actions,
            }],
        })
    }

    fn run_actions(n_pos: u32, actions: &[Value]) -> Result<Vec<Value>> {
        let response = execute(&request(n_pos, actions))?;
        Ok(response["Results"][0]["Actions"]
            .as_array()
            .cloned()
            .unwrap_or_default())
    }

    #[test]
    fn independent_batch_allocator_init_failures_and_autofill_defaults() -> Result<()> {
        let mut single = BatchFixture::new();
        single.add(0, &[0], true)?;
        let mut bad_id_high = BatchFixture::new();
        bad_id_high.add(0, &[4], true)?;
        let mut bad_id_neg = BatchFixture::new();
        bad_id_neg.add(0, &[-1], true)?;
        let token_action = |tok: i32| {
            json!({
                "Op": "init", "EmbeddingWidth": 0, "Embeddings": Value::Null,
                "Tokens": [tok], "Positions": Value::Null, "SequenceIds": Value::Null,
                "Logits": Value::Null, "SequenceLimit": 1, "OutputAll": false,
                "MemoryRanges": Value::Null,
            })
        };
        let init_res = run_actions(
            1,
            &[
                single.init_action(257, None),
                token_action(0),
                token_action(-1),
                bad_id_high.init_action(4, None),
                bad_id_neg.init_action(4, None),
            ],
        )?;
        for entry in init_res {
            assert_eq!(entry["Initialized"], Value::Bool(false));
            assert_eq!(entry["State"], Value::Null);
        }

        let mut four = BatchFixture::new();
        for i in 0..4 {
            four.add(0, &[0], i == 0)?;
        }
        let mut auto_all = four.init_action(4, None);
        auto_all["Positions"] = Value::Null;
        auto_all["SequenceIds"] = Value::Null;
        auto_all["Logits"] = Value::Null;
        let mut mem_override = four.init_action(4, Some(&[(0, 0, 9)]));
        mem_override["Positions"] = Value::Null;
        mem_override["OutputAll"] = Value::Bool(true);

        let res = run_actions(1, &[auto_all, mem_override])?;
        assert_eq!(res[0]["Initialized"], Value::Bool(true));
        assert_eq!(res[0]["State"]["NTokens"], 4);
        assert_eq!(res[0]["State"]["NOutputs"], 1);
        assert_eq!(res[0]["State"]["Batch"]["Positions"], json!([0, 1, 2, 3]));
        assert_eq!(res[0]["State"]["Batch"]["Logits"], json!([0, 0, 0, 1]));
        assert_eq!(res[0]["State"]["SequencePositions"][0], json!({"Min": 0, "Max": 3}));
        assert_eq!(res[0]["State"]["SequencePositions"][1], json!({"Min": -1, "Max": -1}));
        assert_eq!(res[1]["State"]["NOutputs"], 4);
        assert_eq!(res[1]["State"]["Batch"]["Logits"], json!([1, 1, 1, 1]));
        assert_eq!(res[1]["State"]["SequencePositions"][0], json!({"Min": 10, "Max": 12}));
        Ok(())
    }

    #[test]
    fn independent_batch_allocator_position_and_coupling_validation() -> Result<()> {
        let mut non_cont = BatchFixture::new();
        for pos in [0, 1, 3] {
            non_cont.add(pos, &[0], pos == 3)?;
        }
        let mut decreasing = BatchFixture::new();
        for (pos, id) in [4, 5, 0, 1, 6, 2, 3].into_iter().zip([0, 0, 1, 1, 0, 1, 0]) {
            decreasing.add(pos, &[id], false)?;
        }
        let mut equal_pos = BatchFixture::new();
        equal_pos.add(0, &[0], false)?;
        equal_pos.add(0, &[0], false)?;
        equal_pos.add(1, &[0], true)?;
        let mut coupled = BatchFixture::new();
        coupled.add(6, &[0, 1], true)?;
        let mut mem_ok = BatchFixture::new();
        mem_ok.add(10, &[0], false)?;
        mem_ok.add(11, &[0], true)?;
        let mut mem_gap = BatchFixture::new();
        mem_gap.add(11, &[0], false)?;
        mem_gap.add(12, &[0], true)?;
        let mut mem_overlap = BatchFixture::new();
        mem_overlap.add(9, &[0], false)?;
        mem_overlap.add(10, &[0], true)?;

        let res = run_actions(
            1,
            &[
                non_cont.init_action(4, None),
                decreasing.init_action(4, None),
                equal_pos.init_action(4, None),
                coupled.init_action(4, Some(&[(0, 0, 5), (1, 2, 5)])),
                coupled.init_action(4, Some(&[(0, 0, 5), (1, 0, 5)])),
                mem_ok.init_action(4, Some(&[(0, 0, 9)])),
                mem_gap.init_action(4, Some(&[(0, 0, 9)])),
                mem_overlap.init_action(4, Some(&[(0, 0, 9)])),
            ],
        )?;
        let expected = [false, false, true, false, true, true, false, false];
        for (obs, want) in res.iter().zip(expected) {
            assert_eq!(obs["Initialized"], Value::Bool(want));
        }
        Ok(())
    }

    #[test]
    fn independent_batch_allocator_simple_seq_reset_and_reserve() -> Result<()> {
        let mut b = BatchFixture::new();
        b.add_seq(5, 0)?;
        let res = run_actions(
            1,
            &[
                b.init_action(4, None),
                json!({"Op": "split-simple", "Capacity": 2}),
                json!({"Op": "split-simple", "Capacity": 2}),
                json!({"Op": "split-simple", "Capacity": 2}),
                json!({"Op": "split-simple", "Capacity": 2}),
                json!({"Op": "split-reset"}),
                json!({"Op": "split-simple", "Capacity": 10}),
            ],
        )?;
        assert_eq!(res[1]["Ubatch"]["NTokens"], 2);
        assert_eq!(res[1]["Ubatch"]["EqualSequences"], Value::Bool(false));
        assert_eq!(res[1]["Ubatch"]["Embeddings"], json!([0.0, 1.0, 100.0, 101.0]));
        assert_eq!(res[1]["Ubatch"]["SequenceIndices"][0], 0);
        assert_eq!(res[1]["Ubatch"]["SequenceIndices"][1], -1);
        assert_eq!(res[3]["Ubatch"]["NTokens"], 1);
        assert_eq!(res[3]["Ubatch"]["Output"], json!([1]));
        assert_eq!(res[3]["State"]["NUsed"], 5);
        assert_eq!(res[4]["Ubatch"]["NTokens"], 0);
        assert_eq!(res[4]["Ubatch"]["Positions"], Value::Null);
        assert_eq!(res[4]["State"]["OutIds"], json!([4]));
        assert_eq!(res[5]["State"]["NUsed"], 0);
        assert_eq!(res[5]["State"]["OutIds"], json!([]));
        assert_eq!(res[6]["Ubatch"]["NTokens"], 5);

        let mut narrow = BatchFixture::new();
        narrow.add(0, &[0, 1], false)?;
        narrow.add(1, &[0], true)?;
        narrow.add(1, &[1], true)?;
        let seq_res = run_actions(
            1,
            &[
                narrow.init_action(4, None),
                json!({"Op": "split-seq", "Capacity": 8}),
                json!({"Op": "split-seq", "Capacity": 8}),
                json!({"Op": "split-seq", "Capacity": 8}),
                json!({"Op": "reserve", "SequenceTokens": 3, "Sequences": 2}),
            ],
        )?;
        assert_eq!(seq_res[1]["Ubatch"]["NTokens"], 2);
        assert_eq!(seq_res[1]["Ubatch"]["UniqueSequenceIds"], json!([0, 1]));
        assert_eq!(seq_res[2]["Ubatch"]["NTokens"], 1);
        assert_eq!(seq_res[2]["Ubatch"]["UniqueSequenceIds"], json!([1]));
        assert_eq!(seq_res[3]["Ubatch"]["NTokens"], 0);
        assert_eq!(seq_res[4]["Ubatch"]["NTokens"], 6);
        assert_eq!(seq_res[4]["Ubatch"]["HasTokens"], Value::Bool(true));
        assert_eq!(seq_res[4]["Ubatch"]["HasEmbeddings"], Value::Bool(false));
        assert_eq!(seq_res[4]["Ubatch"]["SequenceIds"], json!([[], [], [], [], [], []]));
        assert_eq!(seq_res[4]["State"], Value::Null);
        Ok(())
    }

    #[test]
    fn independent_batch_allocator_equal_split() -> Result<()> {
        let mut unequal = BatchFixture::new();
        unequal.add_seq(4, 0)?;
        unequal.add_seq(2, 1)?;
        let res = run_actions(
            1,
            &[
                unequal.init_action(4, None),
                json!({"Op": "split-equal", "Capacity": 8, "Sequential": false, "KeepTail": 0}),
                json!({"Op": "split-equal", "Capacity": 8, "Sequential": false, "KeepTail": 0}),
                json!({"Op": "split-equal", "Capacity": 8, "Sequential": false, "KeepTail": 0}),
            ],
        )?;
        assert_eq!(res[1]["Ubatch"]["NTokens"], 4);
        assert_eq!(res[1]["Ubatch"]["NSequenceTokens"], 2);
        assert_eq!(res[1]["Ubatch"]["NSequences"], 2);
        assert_eq!(res[1]["Ubatch"]["Positions"], json!([0, 1, 0, 1]));
        assert_eq!(res[2]["Ubatch"]["NTokens"], 2);
        assert_eq!(res[2]["Ubatch"]["Positions"], json!([2, 3]));
        assert_eq!(res[3]["Ubatch"]["NTokens"], 0);

        let mut coupled = BatchFixture::new();
        coupled.add(0, &[0, 1], false)?;
        coupled.add(1, &[0, 1], true)?;
        let cpl_res = run_actions(
            1,
            &[
                coupled.init_action(4, None),
                json!({"Op": "split-equal", "Capacity": 4, "Sequential": true, "KeepTail": 0}),
                json!({"Op": "split-equal", "Capacity": 4, "Sequential": false, "KeepTail": 0}),
            ],
        )?;
        assert_eq!(cpl_res[1]["Ubatch"]["NTokens"], 0);
        assert_eq!(cpl_res[2]["Ubatch"]["NTokens"], 2);
        assert_eq!(cpl_res[2]["Ubatch"]["NSequences"], 1);
        assert_eq!(cpl_res[2]["Ubatch"]["UniqueSequenceIds"], json!([0, 1]));
        Ok(())
    }

    #[test]
    fn independent_batch_allocator_equal_split_keep_tail() -> Result<()> {
        let mut defer = BatchFixture::new();
        defer.add_seq(2, 0)?;
        defer.add_seq(3, 1)?;
        let defer_res = run_actions(
            1,
            &[
                defer.init_action(4, None),
                json!({"Op": "split-equal", "Capacity": 4, "Sequential": true, "KeepTail": 2}),
                json!({"Op": "split-equal", "Capacity": 4, "Sequential": true, "KeepTail": 2}),
            ],
        )?;
        assert_eq!(defer_res[1]["Ubatch"]["NTokens"], 2);
        assert_eq!(defer_res[2]["Ubatch"]["NTokens"], 3);

        let mut violate = BatchFixture::new();
        violate.add_seq(3, 0)?;
        violate.add_seq(3, 1)?;
        let vio_res = run_actions(
            1,
            &[
                violate.init_action(4, None),
                json!({"Op": "split-equal", "Capacity": 4, "Sequential": false, "KeepTail": 2}),
                json!({"Op": "split-equal", "Capacity": 4, "Sequential": false, "KeepTail": 2}),
            ],
        )?;
        assert_eq!(vio_res[1]["Ubatch"]["NTokens"], 3);
        assert_eq!(vio_res[2]["Ubatch"]["NTokens"], 3);

        let mut trunc = BatchFixture::new();
        trunc.add_seq(5, 0)?;
        let trunc_res = run_actions(
            1,
            &[
                trunc.init_action(4, None),
                json!({"Op": "split-equal", "Capacity": 4, "Sequential": false, "KeepTail": 2}),
                json!({"Op": "split-equal", "Capacity": 4, "Sequential": false, "KeepTail": 2}),
            ],
        )?;
        assert_eq!(trunc_res[1]["Ubatch"]["NTokens"], 3);
        assert_eq!(trunc_res[2]["Ubatch"]["NTokens"], 2);
        assert_eq!(trunc_res[2]["Ubatch"]["Positions"], json!([3, 4]));
        Ok(())
    }

    #[test]
    fn independent_batch_allocator_mrope_positions_and_identity() -> Result<()> {
        let mut b = BatchFixture::new();
        b.add(10, &[0], false)?;
        b.add(11, &[0], true)?;
        let mut init_mrope = b.init_action(4, None);
        init_mrope["Positions"] = json!([10, 11, 5, 6, 7, 8, 0, 0]);
        let res = run_actions(
            4,
            &[init_mrope, json!({"Op": "split-simple", "Capacity": 2})],
        )?;
        assert_eq!(res[0]["Initialized"], Value::Bool(true));
        assert_eq!(res[1]["Ubatch"]["NPositions"], 4);
        assert_eq!(res[1]["Ubatch"]["IsPosition2D"], Value::Bool(true));
        assert_eq!(
            res[1]["Ubatch"]["Positions"],
            json!([10, 11, 5, 6, 7, 8, 0, 0])
        );

        let mut no_pos_mrope = b.init_action(4, None);
        no_pos_mrope["Positions"] = Value::Null;
        assert!(execute(&request(4, &[no_pos_mrope])).is_err());

        let mut jump_actions = Vec::new();
        for p0 in [15, 9, 8] {
            let mut single = BatchFixture::new();
            single.add(p0, &[0], true)?;
            let mut action = single.init_action(4, Some(&[(0, 0, 9)]));
            action["Positions"] = json!([p0, 1, 1, 0]);
            jump_actions.push(action);
        }
        let jump_res = run_actions(4, &jump_actions)?;
        assert_eq!(jump_res[0]["Initialized"], Value::Bool(true));
        assert_eq!(jump_res[1]["Initialized"], Value::Bool(true));
        assert_eq!(jump_res[2]["Initialized"], Value::Bool(false));

        let id = identity();
        assert_eq!(id["SupportedSourceIds"], json!([ID]));
        assert_eq!(id["NativeReferenceReplay"], Value::Bool(false));
        assert_eq!(id["ProductionRuntimeCalled"], Value::Bool(false));
        assert_eq!(id["CompleteSourceGate"], Value::Bool(false));
        assert_eq!(id["NativeRetirementQualified"], Value::Bool(false));
        Ok(())
    }

    #[test]
    fn independent_batch_allocator_protocol_bounds_and_refusals() -> Result<()> {
        let mut b = BatchFixture::new();
        b.add(0, &[0], true)?;
        let valid_init = b.init_action(4, None);
        let valid_req = request(1, std::slice::from_ref(&valid_init));
        assert!(handles(&valid_req));
        assert!(!handles(&json!({"Operations": []})));
        assert!(!handles(&json!({"Operations": [{"Kind": "other"}]})));
        assert!(execute(&valid_req).is_ok());

        for bad_version in [json!(0), json!(2), json!(true)] {
            let mut req = valid_req.clone();
            req["SchemaVersion"] = bad_version;
            assert!(execute(&req).is_err());
        }
        for bad_pos in [json!(0), json!(2), json!(3), json!(5), json!(1.0)] {
            let mut req = valid_req.clone();
            req["Operations"][0]["PositionsPerEmbedding"] = bad_pos;
            assert!(execute(&req).is_err());
        }
        for bad_ops in [json!([]), json!(vec![valid_req["Operations"][0].clone(); 129])] {
            let mut req = valid_req.clone();
            req["Operations"] = bad_ops;
            assert!(execute(&req).is_err());
        }
        let mut missing_positions = valid_init.clone();
        missing_positions
            .as_object_mut()
            .ok_or_else(|| super::invalid("test object expected"))?
            .remove("Positions");
        assert!(execute(&request(1, &[missing_positions])).is_err());

        for bad_action in [
            json!({"Op": "unknown"}),
            json!({"Op": "split-simple", "Capacity": 0}),
            json!({"Op": "split-simple", "Capacity": 65}),
            json!({"Op": "split-equal", "Capacity": 2, "Sequential": false, "KeepTail": 2}),
            json!({"Op": "reserve", "SequenceTokens": 16, "Sequences": 5}),
            json!({"Op": "split-reset"}),
            json!({"Op": "split-simple", "Capacity": 2}),
        ] {
            assert!(execute(&request(1, &[bad_action])).is_err());
        }

        let after_reserve = [
            valid_init.clone(),
            json!({"Op": "reserve", "SequenceTokens": 1, "Sequences": 1}),
            json!({"Op": "split-simple", "Capacity": 1}),
        ];
        assert!(execute(&request(1, &after_reserve)).is_err());

        let mut dup_mem = valid_init.clone();
        dup_mem["MemoryRanges"] =
            json!([{"Sequence": 0, "Min": 0, "Max": 1}, {"Sequence": 0, "Min": 0, "Max": 2}]);
        let mut bad_width = valid_init.clone();
        bad_width["EmbeddingWidth"] = json!(0);
        let mut misaligned = valid_init.clone();
        misaligned["Embeddings"] = json!([0.0]);
        let mut bad_seq_len = valid_init;
        bad_seq_len["SequenceIds"] = json!([[0], [0]]);
        for bad in [dup_mem, bad_width, misaligned, bad_seq_len] {
            assert!(execute(&request(1, &[bad])).is_err());
        }
        Ok(())
    }
}
