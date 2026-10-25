//! Owned batch fulfillment and sequence-aware splitting.
//!
//! Preserves Prism `src/llama-batch.cpp` revision
//! `b87ca7225456b06b0975b8239c66b5aecf07e37c` semantics (MIT,
//! PRISM-LICENSE.txt), with explicit errors at malformed native boundaries.

use std::collections::{BTreeMap, BTreeSet};

use crate::{Result, invalid};

/// Native upper bound for distinct sequence IDs.
pub const MAX_SEQUENCES: usize = 256;
const MAX_TOKENS: usize = 1_048_576;

/// Input batch with optional positions, sequence IDs and output flags.
#[derive(Clone, Debug, Default)]
pub struct Input {
    /// Token IDs; mutually exclusive with embeddings.
    pub tokens: Option<Vec<i32>>,
    /// Consecutive embedding rows; mutually exclusive with tokens.
    pub embeddings: Option<Vec<f32>>,
    /// Number of values per embedding row; zero for token input.
    pub embedding_width: usize,
    /// Position sections, each containing one position per input row.
    pub positions: Option<Vec<i32>>,
    /// Nonempty sequence memberships per row; defaults to sequence zero.
    pub sequence_ids: Option<Vec<Vec<i32>>>,
    /// Per-row output flags; defaults to the final row only.
    pub logits: Option<Vec<i8>>,
}

/// Fulfilled batch with independently owned data.
#[derive(Clone, Debug, Default)]
pub struct Batch {
    /// Token IDs, when the input contains tokens.
    pub tokens: Option<Vec<i32>>,
    /// Embedding rows, when the input contains embeddings.
    pub embeddings: Option<Vec<f32>>,
    /// Position sections in section-major order.
    pub positions: Vec<i32>,
    /// Sequence memberships in row order.
    pub sequence_ids: Vec<Vec<i32>>,
    /// Output flags in row order.
    pub logits: Vec<i8>,
}

impl Batch {
    /// Number of token or embedding rows.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.sequence_ids.len()
    }

    /// Whether the batch contains no rows.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.sequence_ids.is_empty()
    }
}

/// Independent microbatch data and sequence layout.
#[derive(Clone, Debug, Default)]
pub struct MicroBatch {
    /// Fulfilled data; absent when splitting has completed.
    pub data: Option<Batch>,
    /// Whether each participating sequence set has the same row count.
    pub equal_sequences: bool,
    /// Rows per sequence set.
    pub sequence_tokens: usize,
    /// Number of participating sequence sets.
    pub sequences: usize,
    /// Position sections per row.
    pub positions: usize,
    /// Distinct sequence IDs in ascending order.
    pub unique_sequence_ids: Vec<usize>,
    /// Mapping from each native sequence ID to its unique-sequence index.
    pub sequence_indices: Vec<i32>,
}

/// Sanitizes a batch and owns its sequence-aware split cursor.
#[derive(Clone, Debug)]
pub struct BatchAllocator {
    positions_per_embedding: usize,
    embedding_width: usize,
    sequence_limit: usize,
    batch: Option<Batch>,
    sequence_positions: Vec<BTreeSet<i32>>,
    sequence_sets: Vec<BTreeSet<usize>>,
    set_indices: BTreeMap<BTreeSet<usize>, Vec<usize>>,
    coupled: bool,
    used: Vec<bool>,
    output_ids: Vec<usize>,
}

impl BatchAllocator {
    /// Creates an allocator for ordinary or four-section image positions.
    ///
    /// # Errors
    /// Rejects a position-section count other than one or four.
    pub fn new(positions_per_embedding: usize) -> Result<Self> {
        if !matches!(positions_per_embedding, 1 | 4) {
            return Err(invalid("batch positions must be one or four"));
        }
        Ok(Self {
            positions_per_embedding,
            embedding_width: 0,
            sequence_limit: 0,
            batch: None,
            sequence_positions: vec![BTreeSet::new(); MAX_SEQUENCES],
            sequence_sets: Vec::new(),
            set_indices: BTreeMap::new(),
            coupled: false,
            used: Vec::new(),
            output_ids: Vec::new(),
        })
    }

    /// Returns the accepted fulfilled batch, absent after a rejected init.
    #[must_use]
    pub const fn batch(&self) -> Option<&Batch> {
        self.batch.as_ref()
    }

    /// Returns native minimum/maximum positions, or `(-1, -1)` for an unused ID.
    #[must_use]
    pub fn sequence_position(&self, id: usize) -> Option<(i32, i32)> {
        self.sequence_positions.get(id).map(|positions| {
            (
                positions.first().copied().unwrap_or(-1),
                positions.last().copied().unwrap_or(-1),
            )
        })
    }

    /// Number of rows already consumed by splits.
    #[must_use]
    pub fn used(&self) -> usize {
        self.used.iter().filter(|used| **used).count()
    }

    /// Original row indices of outputs in split encounter order.
    #[must_use]
    pub fn output_ids(&self) -> &[usize] {
        &self.output_ids
    }

    fn clear(&mut self) {
        self.batch = None;
        self.sequence_sets.clear();
        self.set_indices.clear();
        for positions in &mut self.sequence_positions {
            positions.clear();
        }
        self.used.clear();
        self.output_ids.clear();
        self.coupled = false;
    }

    /// Validates and fulfills a batch, using memory ranges to continue positions.
    ///
    /// `memory` supplies each sequence's previous inclusive minimum/maximum;
    /// missing IDs have `(-1, -1)`. `vocabulary_size` validates token inputs.
    /// `Ok(false)` denotes a semantic rejection (tokens, memberships, continuity,
    /// decreasing positions or diverged coupled histories).
    ///
    /// # Errors
    /// Rejects malformed extents, nonfinite embeddings, an empty batch, more
    /// than 1,048,576 rows, arithmetic overflow or invalid position sections.
    #[allow(
        clippy::too_many_lines,
        reason = "Preserve native fulfillment and semantic checks as one initialization transaction."
    )]
    pub fn init(
        &mut self,
        input: Input,
        vocabulary_size: usize,
        sequence_limit: usize,
        output_all: bool,
        memory: Option<&BTreeMap<usize, (i32, i32)>>,
    ) -> Result<bool> {
        self.clear();
        if sequence_limit == 0 || sequence_limit > MAX_SEQUENCES {
            return Ok(false);
        }
        let count = match (&input.tokens, &input.embeddings) {
            (Some(tokens), None) if input.embedding_width == 0 => tokens.len(),
            (None, Some(embeddings)) if input.embedding_width > 0 => {
                if !embeddings.len().is_multiple_of(input.embedding_width)
                    || embeddings.iter().any(|value| !value.is_finite())
                {
                    return Err(invalid("malformed embedding batch"));
                }
                embeddings.len() / input.embedding_width
            }
            _ => return Err(invalid("batch requires either tokens or embeddings")),
        };
        if count == 0 || count > MAX_TOKENS {
            return Err(invalid("batch row count outside bound"));
        }
        if input.tokens.as_ref().is_some_and(|tokens| {
            tokens
                .iter()
                .any(|token| usize::try_from(*token).map_or(true, |token| token >= vocabulary_size))
        }) {
            return Ok(false);
        }
        let sequence_ids = input.sequence_ids.unwrap_or_else(|| vec![vec![0]; count]);
        if sequence_ids.len() != count || sequence_ids.iter().any(Vec::is_empty) {
            return Err(invalid("batch sequence extent mismatch"));
        }
        if sequence_ids
            .iter()
            .flatten()
            .any(|id| usize::try_from(*id).map_or(true, |id| id >= sequence_limit))
        {
            return Ok(false);
        }
        let memory_range = |id: usize| {
            memory
                .and_then(|ranges| ranges.get(&id))
                .copied()
                .unwrap_or((-1, -1))
        };
        let position_sections = if input.tokens.is_some() {
            1
        } else {
            self.positions_per_embedding
        };
        let positions = if let Some(positions) = input.positions {
            if count.checked_mul(position_sections) != Some(positions.len()) {
                return Err(invalid("batch position extent mismatch"));
            }
            positions
        } else {
            // Missing native positions use the sequential section. Replicate
            // it for embedding sections to keep the owned safe extent complete.
            let mut next = (0..sequence_limit)
                .map(|id| {
                    memory_range(id)
                        .1
                        .checked_add(1)
                        .ok_or_else(|| invalid("batch memory position overflow"))
                })
                .collect::<Result<Vec<_>>>()?;
            let mut first = Vec::with_capacity(count);
            for ids in &sequence_ids {
                let position = next[usize::try_from(ids[0])?];
                first.push(position);
                for &id in ids {
                    next[usize::try_from(id)?] = position
                        .checked_add(1)
                        .ok_or_else(|| invalid("batch position overflow"))?;
                }
            }
            first.repeat(position_sections)
        };
        let mut logits = input.logits.unwrap_or_else(|| {
            let mut flags = vec![i8::from(output_all); count];
            flags[count - 1] = 1;
            flags
        });
        if logits.len() != count {
            return Err(invalid("batch output flag extent mismatch"));
        }
        if output_all {
            logits.fill(1);
        }
        self.embedding_width = input.embedding_width;
        self.sequence_limit = sequence_limit;
        for (index, ids) in sequence_ids.iter().enumerate() {
            let set: BTreeSet<_> = ids
                .iter()
                .map(|id| usize::try_from(*id))
                .collect::<std::result::Result<_, _>>()?;
            self.coupled |= ids.len() > 1;
            for &id in &set {
                self.sequence_positions[id].insert(positions[index]);
            }
            if memory.is_some()
                && ids.iter().skip(1).any(|id| {
                    memory_range(usize::try_from(*id).unwrap_or(MAX_SEQUENCES))
                        != memory_range(usize::try_from(ids[0]).unwrap_or(MAX_SEQUENCES))
                })
            {
                return Ok(false);
            }
            self.set_indices.entry(set.clone()).or_default().push(index);
            self.sequence_sets.push(set);
        }
        for id in 0..sequence_limit {
            let set = &self.sequence_positions[id];
            let (Some(&first), Some(&last)) = (set.first(), set.last()) else {
                continue;
            };
            let previous = memory_range(id).1;
            if self.positions_per_embedding > 1 {
                if previous >= 0
                    && (previous > first || (input.tokens.is_some() && previous == first))
                {
                    return Ok(false);
                }
            } else if (previous >= 0 && previous.checked_add(1) != Some(first))
                || i64::from(last) - i64::from(first) + 1 > i64::try_from(set.len())?
            {
                return Ok(false);
            }
        }
        let mut intersections: Vec<BTreeSet<usize>> =
            vec![(0..sequence_limit).collect(); sequence_limit];
        let mut last = vec![-1; sequence_limit];
        for (index, set) in self.sequence_sets.iter().enumerate() {
            for &id in set {
                intersections[id].retain(|member| set.contains(member));
                if intersections[id].is_empty() || positions[index] < last[id] {
                    return Ok(false);
                }
                last[id] = positions[index];
            }
        }
        self.batch = Some(Batch {
            tokens: input.tokens,
            embeddings: input.embeddings,
            positions,
            sequence_ids,
            logits,
        });
        self.split_reset();
        Ok(true)
    }

    /// Restarts splitting while retaining the fulfilled batch.
    pub fn split_reset(&mut self) {
        self.output_ids.clear();
        self.used = vec![false; self.batch.as_ref().map_or(0, Batch::len)];
    }

    fn check_split(&self, capacity: usize) -> Result<()> {
        if self.batch.is_none() || capacity == 0 || capacity > MAX_TOKENS {
            return Err(invalid(
                "split requires an initialized batch and bounded positive capacity",
            ));
        }
        Ok(())
    }

    fn add(&mut self, indices: &[usize], sequences: usize, equal: bool) -> Result<MicroBatch> {
        let batch = self
            .batch
            .as_ref()
            .ok_or_else(|| invalid("missing accepted batch"))?;
        let mut data = Batch {
            tokens: batch
                .tokens
                .as_ref()
                .map(|tokens| indices.iter().map(|&index| tokens[index]).collect()),
            embeddings: batch.embeddings.as_ref().map(|embeddings| {
                indices
                    .iter()
                    .flat_map(|&index| {
                        &embeddings
                            [index * self.embedding_width..(index + 1) * self.embedding_width]
                    })
                    .copied()
                    .collect()
            }),
            sequence_ids: indices
                .iter()
                .map(|&index| batch.sequence_ids[index].clone())
                .collect(),
            logits: indices.iter().map(|&index| batch.logits[index]).collect(),
            positions: Vec::new(),
        };
        for section in 0..self.positions_per_embedding {
            let offset = if batch.tokens.is_some() {
                0
            } else {
                section * batch.len()
            };
            data.positions
                .extend(indices.iter().map(|&index| batch.positions[offset + index]));
        }
        for &index in indices {
            if batch.logits[index] != 0 {
                self.output_ids.push(index);
            }
        }
        let unique: BTreeSet<_> = indices
            .iter()
            .flat_map(|&index| self.sequence_sets[index].iter().copied())
            .collect();
        let mut sequence_indices = vec![-1; MAX_SEQUENCES];
        for (index, &id) in unique.iter().enumerate() {
            sequence_indices[id] = i32::try_from(index)?;
        }
        Ok(MicroBatch {
            data: Some(data),
            equal_sequences: equal,
            sequence_tokens: indices.len() / sequences,
            sequences,
            positions: self.positions_per_embedding,
            unique_sequence_ids: unique.into_iter().collect(),
            sequence_indices,
        })
    }

    /// Takes the next consecutive rows from the first unused row.
    ///
    /// # Errors
    /// Rejects an uninitialized batch or invalid capacity.
    pub fn split_simple(&mut self, capacity: usize) -> Result<MicroBatch> {
        self.check_split(capacity)?;
        let Some(start) = self.used.iter().position(|used| !used) else {
            return Ok(MicroBatch::default());
        };
        let indices: Vec<_> = (start..self.used.len()).take(capacity).collect();
        for &index in &indices {
            self.used[index] = true;
        }
        self.add(&indices, indices.len(), false)
    }

    /// Takes one sequence set and subsequent sets that are its subsets.
    ///
    /// # Errors
    /// Rejects an uninitialized batch or invalid capacity.
    pub fn split_sequence(&mut self, capacity: usize) -> Result<MicroBatch> {
        self.check_split(capacity)?;
        let Some(start) = self.used.iter().position(|used| !used) else {
            return Ok(MicroBatch::default());
        };
        let mut current = self.sequence_sets[start].clone();
        let mut indices = Vec::new();
        for index in start..self.used.len() {
            if self.used[index] || !self.sequence_sets[index].is_subset(&current) {
                continue;
            }
            indices.push(index);
            self.used[index] = true;
            current.clone_from(&self.sequence_sets[index]);
            if indices.len() == capacity {
                break;
            }
        }
        self.add(&indices, 1, true)
    }

    /// Splits equal-length, non-overlapping sequence sets, preserving a tail.
    ///
    /// Sequential mode requires consecutive first sequence IDs and rejects
    /// coupled input by returning an empty microbatch. `keep_tail` prevents a
    /// sequence's final tokens from being split across microbatches.
    ///
    /// # Errors
    /// Rejects an uninitialized batch, invalid capacity, or a nonzero tail at
    /// least as large as capacity.
    #[allow(
        clippy::too_many_lines,
        reason = "Preserve the native equal-split and keep-tail transaction as one operation."
    )]
    pub fn split_equal(
        &mut self,
        capacity: usize,
        sequential: bool,
        keep_tail: usize,
    ) -> Result<MicroBatch> {
        self.check_split(capacity)?;
        if keep_tail >= capacity && keep_tail != 0 {
            return Err(invalid("batch tail must be smaller than capacity"));
        }
        if sequential && self.coupled {
            return Ok(MicroBatch::default());
        }
        let mut sets: Vec<BTreeSet<usize>> = Vec::new();
        let mut last = None;
        for (index, set) in self.sequence_sets.iter().enumerate() {
            if self.used[index] {
                continue;
            }
            let first = usize::try_from(
                self.batch
                    .as_ref()
                    .ok_or_else(|| invalid("missing batch"))?
                    .sequence_ids[index][0],
            )?;
            if sets.iter().all(|other| other.is_disjoint(set))
                && (!sequential || last.is_none_or(|last| first == last + 1))
            {
                sets.push(set.clone());
                last = Some(first);
                if sets.len() > capacity {
                    break;
                }
            }
        }
        if sets.is_empty() {
            return Ok(MicroBatch::default());
        }
        let lists: Vec<_> = sets.iter().map(|set| &self.set_indices[set]).collect();
        let mut cursors: Vec<_> = lists
            .iter()
            .map(|indices| {
                indices
                    .iter()
                    .position(|&index| !self.used[index])
                    .unwrap_or(indices.len())
            })
            .collect();
        let mut selected = vec![Vec::new(); sets.len()];
        loop {
            if lists
                .iter()
                .zip(&cursors)
                .any(|(list, cursor)| *cursor >= list.len())
            {
                break;
            }
            for sequence in 0..sets.len() {
                let index = lists[sequence][cursors[sequence]];
                selected[sequence].push(index);
                self.used[index] = true;
                cursors[sequence] += 1;
            }
            if (selected[0].len() + 1) * sets.len() > capacity {
                break;
            }
        }
        let mut keep = sets.len();
        if keep_tail > 0 {
            keep = lists
                .iter()
                .zip(&cursors)
                .take_while(|(list, cursor)| {
                    list.len() == **cursor || list.len() - **cursor >= keep_tail
                })
                .count();
            if keep == 0 {
                if selected[0].len() + lists[0].len() - cursors[0] <= capacity {
                    while cursors[0] < lists[0].len() {
                        let index = lists[0][cursors[0]];
                        selected[0].push(index);
                        self.used[index] = true;
                        cursors[0] += 1;
                    }
                } else {
                    while lists[0].len() - cursors[0] < keep_tail {
                        let index = selected[0]
                            .pop()
                            .ok_or_else(|| invalid("cannot preserve batch tail"))?;
                        self.used[index] = false;
                        cursors[0] -= 1;
                    }
                }
                keep = 1;
            }
            for deferred in &selected[keep..] {
                for &index in deferred {
                    self.used[index] = false;
                }
            }
        }
        let indices: Vec<_> = selected[..keep].iter().flatten().copied().collect();
        self.add(&indices, keep, true)
    }

    /// Reserves a defined zero-filled token microbatch and clears active input.
    ///
    /// # Errors
    /// Rejects zero sequence dimensions, more than 256 sequences, overflow, or
    /// a reservation larger than 1,048,576 rows.
    pub fn reserve(&mut self, sequence_tokens: usize, sequences: usize) -> Result<MicroBatch> {
        let count = sequence_tokens
            .checked_mul(sequences)
            .filter(|&count| count > 0 && count <= MAX_TOKENS)
            .ok_or_else(|| invalid("batch reservation row count outside bound"))?;
        if sequence_tokens == 0 || sequences == 0 || sequences > MAX_SEQUENCES {
            return Err(invalid("batch reservation sequence count outside bound"));
        }
        self.clear();
        let mut sequence_indices = vec![-1; MAX_SEQUENCES];
        for (index, value) in sequence_indices.iter_mut().take(sequences).enumerate() {
            *value = i32::try_from(index)?;
        }
        Ok(MicroBatch {
            data: Some(Batch {
                tokens: Some(vec![0; count]),
                embeddings: None,
                positions: vec![0; count * self.positions_per_embedding],
                sequence_ids: vec![Vec::new(); count],
                logits: vec![0; count],
            }),
            equal_sequences: true,
            sequence_tokens,
            sequences,
            positions: self.positions_per_embedding,
            unique_sequence_ids: (0..sequences).collect(),
            sequence_indices,
        })
    }
}
