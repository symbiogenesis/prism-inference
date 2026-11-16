//! Ranked GPT-2 byte-pair merges for **already pretokenized byte pieces**.
//!
//! This is only the merge core. It does not implement the Qwen35 regex splitter,
//! chat templates, special-token recognition, or an arbitrary-text tokenizer.
//! Callers must supply the exact pretokenizer boundaries: merges never cross
//! those boundaries. Only normal vocabulary IDs can be emitted.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

use bonsai_gguf::{Metadata, MetadataArray};

use crate::vocabulary::{MAX_PIECE_BYTES, ValidatedVocabulary, decode_character};
use crate::{Result, invalid};

/// Maximum number of ordered merge rules admitted from metadata.
pub const MAX_BPE_MERGES: usize = 1_000_000;
/// Maximum aggregate UTF-8 bytes in serialized merge rules.
pub const MAX_BPE_MERGE_BYTES: usize = 64 * 1024 * 1024;
/// Maximum bytes in one already pretokenized input piece.
pub const MAX_BPE_PIECE_BYTES: usize = 64 * 1024;
/// Maximum total input bytes admitted by [`Bpe::encode_pieces`].
pub const MAX_BPE_INPUT_BYTES: usize = 1024 * 1024;
/// Maximum pretokenized pieces admitted in one call, including empty pieces.
pub const MAX_BPE_INPUT_PIECES: usize = 64 * 1024;

const EMPTY_RANK: u32 = u32::MAX;

const _: () = assert!(MAX_BPE_MERGES < 0xFFFF_FFFF);

#[derive(Clone, Copy, Debug)]
struct Merge {
    rank: u32,
    token: u32,
}

#[derive(Clone, Copy, Debug)]
struct Slot {
    pair: u64,
    rank: u32,
    token: u32,
}

const _: () = assert!(size_of::<Slot>() == 16);

impl Default for Slot {
    fn default() -> Self {
        Self {
            pair: 0,
            rank: EMPTY_RANK,
            token: 0,
        }
    }
}

fn pack_pair(left: u32, right: u32) -> u64 {
    (u64::from(left) << 32) | u64::from(right)
}

const fn mix_pair(mut key: u64) -> u64 {
    key = (key ^ (key >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    key = (key ^ (key >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    key ^ (key >> 31)
}

fn slot_index(hash: u64, mask: usize) -> usize {
    u64::try_from(mask).map_or(0, |mask_u64| usize::try_from(hash & mask_u64).unwrap_or(0))
}

#[derive(Debug)]
struct MergeTable {
    slots: Vec<Slot>,
    len: usize,
}

impl MergeTable {
    fn with_capacity(merges: usize) -> Result<Self> {
        if merges > MAX_BPE_MERGES {
            return Err(invalid("BPE merge count exceeds supported bounds"));
        }
        if merges == 0 {
            return Ok(Self {
                slots: Vec::new(),
                len: 0,
            });
        }
        let capacity = merges
            .checked_mul(2)
            .ok_or_else(|| invalid("BPE merge table capacity overflow"))?
            .max(8)
            .next_power_of_two();
        let mut slots = Vec::new();
        slots.try_reserve_exact(capacity)?;
        slots.resize(capacity, Slot::default());
        Ok(Self { slots, len: 0 })
    }

    fn rehash(&mut self, new_capacity: usize) -> Result<()> {
        let mut new_slots = Vec::new();
        new_slots.try_reserve_exact(new_capacity)?;
        new_slots.resize(new_capacity, Slot::default());
        let mask = new_capacity - 1;
        for &slot in &self.slots {
            if slot.rank == EMPTY_RANK {
                continue;
            }
            let mut index = slot_index(mix_pair(slot.pair), mask);
            while new_slots[index].rank != EMPTY_RANK {
                index = (index + 1) & mask;
            }
            new_slots[index] = slot;
        }
        self.slots = new_slots;
        Ok(())
    }

    fn insert(&mut self, left: u32, right: u32, rank: u32, token: u32) -> Result<Option<Merge>> {
        if rank == EMPTY_RANK || usize::try_from(rank)? >= MAX_BPE_MERGES {
            return Err(invalid("BPE merge rank exceeds supported bounds"));
        }
        let pair = pack_pair(left, right);
        if !self.slots.is_empty() {
            let slot_count = self.slots.len();
            let mask = slot_count - 1;
            let mut index = slot_index(mix_pair(pair), mask);
            loop {
                let slot = &mut self.slots[index];
                if slot.rank == EMPTY_RANK {
                    if self.len >= MAX_BPE_MERGES {
                        return Err(invalid("BPE merge count exceeds supported bounds"));
                    }
                    if (self.len + 1) * 2 <= slot_count {
                        *slot = Slot { pair, rank, token };
                        self.len += 1;
                        return Ok(None);
                    }
                    break;
                }
                if slot.pair == pair {
                    let previous = Merge {
                        rank: slot.rank,
                        token: slot.token,
                    };
                    slot.rank = rank;
                    slot.token = token;
                    return Ok(Some(previous));
                }
                index = (index + 1) & mask;
            }
        } else if self.len >= MAX_BPE_MERGES {
            return Err(invalid("BPE merge count exceeds supported bounds"));
        }
        let new_capacity = self
            .slots
            .len()
            .max(4)
            .checked_mul(2)
            .ok_or_else(|| invalid("BPE merge table capacity overflow"))?;
        self.rehash(new_capacity)?;
        let mask = self.slots.len() - 1;
        let mut index = slot_index(mix_pair(pair), mask);
        while self.slots[index].rank != EMPTY_RANK {
            index = (index + 1) & mask;
        }
        self.slots[index] = Slot { pair, rank, token };
        self.len += 1;
        Ok(None)
    }

    fn get(&self, left: u32, right: u32) -> Option<Merge> {
        if self.slots.is_empty() {
            return None;
        }
        let pair = pack_pair(left, right);
        let mask = self.slots.len() - 1;
        let mut index = slot_index(mix_pair(pair), mask);
        loop {
            let slot = self.slots[index];
            if slot.rank == EMPTY_RANK {
                return None;
            }
            if slot.pair == pair {
                return Some(Merge {
                    rank: slot.rank,
                    token: slot.token,
                });
            }
            index = (index + 1) & mask;
        }
    }
}

#[derive(Debug)]
struct Node {
    token: u32,
    previous: Option<usize>,
    next: Option<usize>,
    alive: bool,
}

// Rank first, original left position second: equal-ranked overlapping pairs
// deterministically merge at the leftmost surviving position.
type Candidate = (u32, usize, usize, u32, u32, u32);

const PIECE_CACHE_SLOTS: usize = 16;
const PIECE_CACHE_BYTES: usize = 32;
// Keep common uncached words off the linked-node/heap path. The bounded scan
// performs at most 120 pair comparisons and uses no retained scratch storage.
const SHORT_PIECE_BYTES: usize = 16;

#[derive(Clone, Copy, Debug, Default)]
struct CachedPiece {
    bytes: [u8; PIECE_CACHE_BYTES],
    tokens: [u32; PIECE_CACHE_BYTES],
    byte_len: usize,
    token_len: usize,
}

// Owned by one encoding request: immutable BPE rules remain freely shareable.
// Clearing between pieces retains capacity bounded by the largest piece seen.
// The small inline cache never allocates or escapes this request. Retaining a
// borrowed owner makes address equality safe even if callers change rule sets:
// the original BPE cannot be dropped while this scratch retains its identity.
#[derive(Debug, Default)]
pub(crate) struct BpeScratch<'a> {
    nodes: Vec<Node>,
    candidates: BinaryHeap<Reverse<Candidate>>,
    owner: Option<&'a Bpe>,
    pieces: [CachedPiece; PIECE_CACHE_SLOTS],
    #[cfg(test)]
    uncached_pieces: usize,
}

impl<'a> BpeScratch<'a> {
    fn cached(&mut self, owner: &'a Bpe, bytes: &[u8]) -> Option<&[u32]> {
        if self
            .owner
            .is_some_and(|previous| !std::ptr::eq(previous, owner))
        {
            self.pieces.fill(CachedPiece::default());
        }
        self.owner = Some(owner);
        if bytes.len() > PIECE_CACHE_BYTES {
            return None;
        }
        let entry = &self.pieces[piece_slot(bytes)];
        (entry.byte_len == bytes.len() && entry.bytes[..entry.byte_len] == *bytes)
            .then_some(&entry.tokens[..entry.token_len])
    }

    fn retain(&mut self, bytes: &[u8], tokens: &[u32]) {
        if bytes.len() > PIECE_CACHE_BYTES {
            return;
        }
        // Each byte begins as one token and merging only reduces that count.
        let entry = &mut self.pieces[piece_slot(bytes)];
        entry.bytes[..bytes.len()].copy_from_slice(bytes);
        entry.tokens[..tokens.len()].copy_from_slice(tokens);
        entry.byte_len = bytes.len();
        entry.token_len = tokens.len();
    }
}

fn piece_slot(bytes: &[u8]) -> usize {
    // Mask each bounded step: multiplication cannot overflow even for hostile
    // pieces. Full byte equality, never this hash, determines cache identity.
    bytes.iter().fold(0, |slot, &byte| {
        (slot * 31 + usize::from(byte)) & (PIECE_CACHE_SLOTS - 1)
    })
}

/// Bounded normal-token BPE rules, independent of pretokenization and templates.
///
/// Every byte starts as its normal single-byte vocabulary ID. At each step the
/// lowest-ranked adjacent rule wins, with leftmost ties. Multi-byte vocabulary
/// entries without a matching merge path are never selected by longest match.
#[derive(Debug)]
pub struct Bpe {
    bytes: [Option<u32>; 256],
    merges: MergeTable,
}

impl Bpe {
    /// Validates the pinned `gpt2` / `qwen35` vocabulary and ordered merge array.
    ///
    /// `tokenizer.ggml.merges` must be a GGUF `STRING` array. Each entry is two
    /// nonempty serialized GPT-2 alphabet pieces separated by exactly one ASCII
    /// space. Both operands and their concatenation must be normal vocabulary
    /// tokens. Array position is the rank; duplicate pairs and duplicate normal
    /// vocabulary spellings are rejected. An empty merge array is valid.
    ///
    /// Missing normal single-byte tokens are allowed at construction, but any
    /// input using such a byte fails. Control, user-defined and byte-type tokens
    /// cannot participate in ordinary byte encoding, even with matching text.
    /// Metadata bounds are checked before allocating merge storage. Retained
    /// storage is a 256-entry byte table and at most [`MAX_BPE_MERGES`] rules.
    /// Construction validates all vocabulary pieces without decoding them into
    /// retained storage. It uses a borrowed normal-token lookup table and one
    /// concatenated rule string of at most `2 * MAX_PIECE_BYTES` UTF-8 bytes
    /// (the alphabet uses at most two UTF-8 bytes per decoded byte).
    ///
    /// # Errors
    /// Rejects invalid vocabulary metadata, malformed or missing merges, missing
    /// normal operands/results, duplicates, resource limits and allocation errors.
    pub fn from_metadata(metadata: &Metadata, vocab_size: usize) -> Result<Self> {
        Self::from_metadata_with_vocabulary(metadata, vocab_size).map(|(bpe, _)| bpe)
    }

    // Composed tokenizers reuse this checked view for special spellings, while
    // preserving merge bounds before vocabulary validation and merge allocation.
    pub(crate) fn from_metadata_with_vocabulary(
        metadata: &Metadata,
        vocab_size: usize,
    ) -> Result<(Self, ValidatedVocabulary<'_>)> {
        let Some(MetadataArray::String(rules)) = metadata
            .get("tokenizer.ggml.merges")
            .and_then(|value| value.as_array())
        else {
            return Err(invalid("BPE requires a STRING merge array"));
        };
        if rules.len() > MAX_BPE_MERGES {
            return Err(invalid("BPE merge count exceeds supported bounds"));
        }
        let mut rule_bytes = 0_usize;
        for rule in rules {
            if rule.len() > 2 * MAX_PIECE_BYTES + 1 {
                return Err(invalid("BPE merge exceeds serialized rule byte limit"));
            }
            rule_bytes = rule_bytes
                .checked_add(rule.len())
                .filter(|&length| length <= MAX_BPE_MERGE_BYTES)
                .ok_or_else(|| invalid("BPE merges exceed serialized byte limit"))?;
        }
        let vocabulary = ValidatedVocabulary::from_metadata(metadata, vocab_size)?;
        let mut normal = HashMap::new();
        normal.try_reserve(vocabulary.len())?;
        let mut bytes = [None; 256];
        for (index, (token, kind)) in vocabulary.entries().enumerate() {
            if kind != 1 {
                continue;
            }
            let id = u32::try_from(index)?;
            if normal.insert(token, id).is_some() {
                return Err(invalid("BPE has duplicate normal vocabulary pieces"));
            }
            let mut characters = token.chars();
            if let (Some(character), None) = (characters.next(), characters.next()) {
                bytes[usize::from(decode_character(character)?)] = Some(id);
            }
        }
        let mut merges = MergeTable::with_capacity(rules.len())?;
        let mut combined = String::new();
        for (rank, rule) in rules.iter().enumerate() {
            let Some((left, right)) = rule.split_once(' ') else {
                return Err(invalid("BPE merge requires one ASCII space separator"));
            };
            if left.is_empty() || right.is_empty() || right.contains(' ') {
                return Err(invalid("BPE merge has empty operands or extra separators"));
            }
            let left_id = *normal
                .get(left)
                .ok_or_else(|| invalid("BPE merge left operand is not a normal token"))?;
            let right_id = *normal
                .get(right)
                .ok_or_else(|| invalid("BPE merge right operand is not a normal token"))?;
            combined.clear();
            combined.try_reserve(rule.len() - 1)?;
            combined.push_str(left);
            combined.push_str(right);
            let token = *normal
                .get(combined.as_str())
                .ok_or_else(|| invalid("BPE merge result is not a normal token"))?;
            let rank = u32::try_from(rank)?;
            if merges.insert(left_id, right_id, rank, token)?.is_some() {
                return Err(invalid("BPE has duplicate merge pairs"));
            }
        }
        Ok((Self { bytes, merges }, vocabulary))
    }

    /// Encodes one **already pretokenized** byte piece using ranked merges.
    ///
    /// No special tokens are recognized or inserted. Arbitrary UTF-8 and binary
    /// bytes are preserved; callers can recover the bytes with
    /// [`Vocabulary`](crate::vocabulary::Vocabulary).
    /// Pieces of at most sixteen bytes merge in bounded stack storage without
    /// node or candidate allocations. Larger pieces retain the heap path.
    /// The 64 KiB piece bound limits work to fewer than `3 * bytes.len()` heap
    /// insertions and `O(n log n)` comparisons, including stale candidates.
    /// Scratch allocations reserve for at most `n` nodes, `3 * n` candidates and
    /// `n` output IDs, independently of the untrusted caller's token limit.
    ///
    /// # Errors
    /// Rejects oversized pieces, bytes missing normal single-byte vocabulary
    /// tokens, final outputs above `max_tokens`, and allocation failures.
    pub fn encode_piece(&self, bytes: &[u8], max_tokens: usize) -> Result<Vec<u32>> {
        let mut output = Vec::new();
        self.encode_piece_into(bytes, max_tokens, &mut output, &mut BpeScratch::default())?;
        Ok(output)
    }

    // Append only after the complete piece passes validation. Callers composing
    // pieces can share output storage without allocating and copying each result.
    pub(crate) fn encode_piece_into<'a>(
        &'a self,
        bytes: &[u8],
        max_tokens: usize,
        output: &mut Vec<u32>,
        scratch: &mut BpeScratch<'a>,
    ) -> Result<()> {
        if bytes.len() > MAX_BPE_PIECE_BYTES {
            return Err(invalid("BPE input piece exceeds byte limit"));
        }
        if bytes.is_empty() {
            return Ok(());
        }
        if max_tokens == 0 {
            // Missing byte vocabulary is corruption, including when no output
            // capacity remains. Preserve that classification before the bound.
            if bytes
                .iter()
                .any(|&byte| self.bytes[usize::from(byte)].is_none())
            {
                return Err(invalid("BPE input byte has no normal single-byte token"));
            }
            return Err(crate::TokenLimitExceeded.into());
        }
        if let Some(tokens) = scratch.cached(self, bytes) {
            if tokens.len() > max_tokens {
                return Err(crate::TokenLimitExceeded.into());
            }
            output.try_reserve_exact(tokens.len())?;
            output.extend_from_slice(tokens);
            return Ok(());
        }
        let start = output.len();
        self.encode_uncached(bytes, max_tokens, output, scratch)?;
        scratch.retain(bytes, &output[start..]);
        Ok(())
    }

    fn encode_uncached(
        &self,
        bytes: &[u8],
        max_tokens: usize,
        output: &mut Vec<u32>,
        scratch: &mut BpeScratch<'_>,
    ) -> Result<()> {
        #[cfg(test)]
        {
            scratch.uncached_pieces += 1;
        }
        if bytes.len() <= SHORT_PIECE_BYTES {
            return self.encode_short(bytes, max_tokens, output);
        }
        let BpeScratch {
            nodes, candidates, ..
        } = scratch;
        nodes.clear();
        candidates.clear();
        nodes.try_reserve_exact(bytes.len())?;
        for (index, &byte) in bytes.iter().enumerate() {
            let token = self.bytes[usize::from(byte)]
                .ok_or_else(|| invalid("BPE input byte has no normal single-byte token"))?;
            nodes.push(Node {
                token,
                previous: index.checked_sub(1),
                next: (index + 1 < bytes.len()).then_some(index + 1),
                alive: true,
            });
        }
        candidates.try_reserve_exact(bytes.len() * 3)?;
        for left in 0..bytes.len() - 1 {
            if let Some(candidate) = self.candidate(nodes, left) {
                candidates.push(candidate);
            }
        }
        let mut remaining = nodes.len();
        while let Some(Reverse((_, left, right, left_token, right_token, token))) = candidates.pop()
        {
            if !nodes[left].alive
                || !nodes[right].alive
                || nodes[left].next != Some(right)
                || nodes[left].token != left_token
                || nodes[right].token != right_token
            {
                continue;
            }
            let next = nodes[right].next;
            nodes[left].token = token;
            nodes[left].next = next;
            nodes[right].alive = false;
            if let Some(next) = next {
                nodes[next].previous = Some(left);
            }
            remaining -= 1;
            if let Some(previous) = nodes[left].previous
                && let Some(candidate) = self.candidate(nodes, previous)
            {
                candidates.push(candidate);
            }
            if let Some(candidate) = self.candidate(nodes, left) {
                candidates.push(candidate);
            }
        }
        if remaining > max_tokens {
            return Err(crate::TokenLimitExceeded.into());
        }
        output.try_reserve_exact(remaining)?;
        let mut current = Some(0);
        while let Some(index) = current {
            output.push(nodes[index].token);
            current = nodes[index].next;
        }
        Ok(())
    }

    // Cache each live adjacency once, then refresh only the two pairs changed
    // by a merge. Compact order is original left-position order, so selecting
    // the first minimum rank preserves heap rank/leftmost ties exactly. Unlike
    // a complete rescan oracle, unchanged pairs need no repeated hash lookup.
    fn encode_short(&self, bytes: &[u8], max_tokens: usize, output: &mut Vec<u32>) -> Result<()> {
        let mut tokens = [0; SHORT_PIECE_BYTES];
        for (token, &byte) in tokens.iter_mut().zip(bytes) {
            *token = self.bytes[usize::from(byte)]
                .ok_or_else(|| invalid("BPE input byte has no normal single-byte token"))?;
        }
        let mut pairs = [None; SHORT_PIECE_BYTES - 1];
        for (pair, adjacent) in pairs.iter_mut().zip(tokens[..bytes.len()].windows(2)) {
            *pair = self.merges.get(adjacent[0], adjacent[1]);
        }
        let mut remaining = bytes.len();
        loop {
            let next = pairs[..remaining - 1]
                .iter()
                .enumerate()
                .filter_map(|(left, pair)| pair.map(|merge| (merge.rank, left, merge.token)))
                .min_by_key(|&(rank, left, _)| (rank, left));
            let Some((_, left, token)) = next else {
                break;
            };
            tokens[left] = token;
            tokens.copy_within(left + 2..remaining, left + 1);
            if left + 2 < remaining {
                pairs.copy_within(left + 2..remaining - 1, left + 1);
            }
            remaining -= 1;
            if left != 0 {
                pairs[left - 1] = self.merges.get(tokens[left - 1], tokens[left]);
            }
            if left + 1 < remaining {
                pairs[left] = self.merges.get(tokens[left], tokens[left + 1]);
            }
        }
        if remaining > max_tokens {
            return Err(crate::TokenLimitExceeded.into());
        }
        output.try_reserve_exact(remaining)?;
        output.extend_from_slice(&tokens[..remaining]);
        Ok(())
    }

    /// Encodes supplied pretokenizer boundaries without merging across pieces.
    ///
    /// This does not find the boundaries itself. At most 64 Ki pieces and 1 MiB
    /// total bytes are admitted, with the same per-piece bound as
    /// [`Self::encode_piece`]. Empty pieces contribute no output.
    /// Node and candidate storage is reused across pieces, with capacity bounded
    /// by the largest piece in this request. Up to sixteen exact pieces of at
    /// most 32 bytes reuse completed token IDs in fixed request-local storage;
    /// collisions replace entries and larger pieces bypass the cache. The cache
    /// adds 2,824 bytes on a 64-bit target, including its borrowed rule identity.
    ///
    /// # Errors
    /// Rejects input bounds, missing normal byte tokens, aggregate outputs above
    /// `max_tokens`, and allocation failures. No partial output is returned.
    pub fn encode_pieces(&self, pieces: &[&[u8]], max_tokens: usize) -> Result<Vec<u32>> {
        if pieces.len() > MAX_BPE_INPUT_PIECES {
            return Err(invalid("BPE input piece count exceeds limit"));
        }
        let mut total_bytes = 0_usize;
        for piece in pieces {
            if piece.len() > MAX_BPE_PIECE_BYTES {
                return Err(invalid("BPE input piece exceeds byte limit"));
            }
            total_bytes = total_bytes
                .checked_add(piece.len())
                .filter(|&length| length <= MAX_BPE_INPUT_BYTES)
                .ok_or_else(|| invalid("BPE total input exceeds byte limit"))?;
        }
        let mut output = Vec::new();
        output.try_reserve_exact(total_bytes.min(max_tokens))?;
        let mut scratch = BpeScratch::default();
        for piece in pieces {
            self.encode_piece_into(piece, max_tokens - output.len(), &mut output, &mut scratch)?;
        }
        Ok(output)
    }

    fn candidate(&self, nodes: &[Node], left: usize) -> Option<Reverse<Candidate>> {
        let right = nodes[left].next?;
        let left_token = nodes[left].token;
        let right_token = nodes[right].token;
        let merge = self.merges.get(left_token, right_token)?;
        Some(Reverse((
            merge.rank,
            left,
            right,
            left_token,
            right_token,
            merge.token,
        )))
    }
}

#[cfg(test)]
mod piece_cache_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use bonsai_gguf::MetadataValue;

    #[test]
    fn merge_table_handles_hash_collisions_growth_and_updates() -> Result<()> {
        // Find three distinct packed pairs that map to the exact same slot mod 8.
        let mut table = MergeTable::with_capacity(4)?;
        assert_eq!(table.slots.len(), 8);
        let mask = table.slots.len() - 1;
        let target_slot = slot_index(mix_pair(pack_pair(10, 20)), mask);
        let mut colliding = vec![(10_u32, 20_u32)];
        for right in 0_u32..512 {
            if right == 20 {
                continue;
            }
            if slot_index(mix_pair(pack_pair(10, right)), mask) == target_slot {
                colliding.push((10, right));
                if colliding.len() == 3 {
                    break;
                }
            }
        }
        assert_eq!(colliding.len(), 3);

        for (rank, &(left, right)) in colliding.iter().enumerate() {
            assert!(
                table
                    .insert(
                        left,
                        right,
                        u32::try_from(rank)?,
                        1000 + u32::try_from(rank)?
                    )?
                    .is_none()
            );
        }
        // Add boundary pairs to ensure high/low 32-bit packing never aliases, forcing a rehash past 4 entries.
        let extra = [
            (0_u32, u32::MAX),
            (u32::MAX, 0_u32),
            (u32::MAX, u32::MAX),
            (1_u32, 0_u32),
            (0_u32, 1_u32),
        ];
        for (offset, &(left, right)) in extra.iter().enumerate() {
            let rank = u32::try_from(colliding.len() + offset)?;
            assert!(table.insert(left, right, rank, 2000 + rank)?.is_none());
        }
        assert_eq!(table.len, 8);
        assert_eq!(table.slots.len(), 16);

        for (rank, &(left, right)) in colliding.iter().enumerate() {
            let found = table
                .get(left, right)
                .ok_or_else(|| invalid("missing colliding entry"))?;
            assert_eq!(found.rank, u32::try_from(rank)?);
            assert_eq!(found.token, 1000 + u32::try_from(rank)?);
        }
        for (offset, &(left, right)) in extra.iter().enumerate() {
            let rank = u32::try_from(colliding.len() + offset)?;
            let found = table
                .get(left, right)
                .ok_or_else(|| invalid("missing boundary entry"))?;
            assert_eq!(found.rank, rank);
            assert_eq!(found.token, 2000 + rank);
        }
        assert!(table.get(999, 999).is_none());

        let replaced = table
            .insert(colliding[1].0, colliding[1].1, 42, 7777)?
            .ok_or_else(|| invalid("expected existing merge"))?;
        assert_eq!(replaced.rank, 1);
        assert_eq!(replaced.token, 1001);
        let updated = table
            .get(colliding[1].0, colliding[1].1)
            .ok_or_else(|| invalid("missing updated entry"))?;
        assert_eq!(updated.rank, 42);
        assert_eq!(updated.token, 7777);
        assert_eq!(table.len, 8);
        Ok(())
    }

    #[test]
    fn merge_table_enforces_capacity_and_rank_overflow_guards() -> Result<()> {
        assert!(MergeTable::with_capacity(MAX_BPE_MERGES + 1).is_err());

        let mut empty = MergeTable::with_capacity(0)?;
        assert!(empty.get(1, 2).is_none());
        assert!(empty.insert(1, 2, EMPTY_RANK, 3).is_err());
        assert!(
            empty
                .insert(1, 2, u32::try_from(MAX_BPE_MERGES)?, 3)
                .is_err()
        );
        assert!(empty.insert(1, 2, 0, 3)?.is_none());
        assert_eq!(empty.get(1, 2).map(|m| (m.rank, m.token)), Some((0, 3)));

        // Simulate saturated MAX_BPE_MERGES count without allocating a million entries:
        // updating an existing entry still succeeds, while adding a new pair is rejected.
        empty.len = MAX_BPE_MERGES;
        assert!(empty.insert(1, 2, 1, 4)?.is_some());
        assert!(empty.insert(9, 10, 1, 5).is_err());
        Ok(())
    }

    #[test]
    fn bpe_open_addressed_merge_table_preserves_encoding_parity() -> Result<()> {
        let tokens = [
            ("a", 1),
            ("b", 1),
            ("c", 1),
            ("ab", 1),
            ("bc", 1),
            ("aa", 1),
            ("abc", 1),
            ("abca", 1),
        ];
        let metadata = Metadata::from([
            ("tokenizer.ggml.model".to_owned(), "gpt2".into()),
            ("tokenizer.ggml.pre".to_owned(), "qwen35".into()),
            (
                "tokenizer.ggml.tokens".to_owned(),
                tokens
                    .iter()
                    .map(|(text, _)| *text)
                    .collect::<Vec<_>>()
                    .into(),
            ),
            (
                "tokenizer.ggml.token_type".to_owned(),
                tokens
                    .iter()
                    .map(|(_, kind)| *kind)
                    .collect::<Vec<_>>()
                    .into(),
            ),
            (
                "tokenizer.ggml.merges".to_owned(),
                MetadataValue::from(vec!["a b", "b c", "ab c", "a a", "abc a"]),
            ),
        ]);
        let bpe = Bpe::from_metadata(&metadata, tokens.len())?;
        // Short piece path (<= 16 bytes) and heap/node path (> 16 bytes) agree on merge order.
        // At index 15..18 (`...abc a a a`), rank 3 ("a a") precedes rank 4 ("abc a").
        assert_eq!(bpe.encode_piece(b"abca", 4)?, [7]);
        assert_eq!(bpe.encode_piece(b"aaa", 3)?, [5, 0]);
        let long_piece = b"abcaabcaabcaabcaaa";
        assert_eq!(bpe.encode_piece(long_piece, 10)?, [7, 7, 7, 6, 5, 0]);
        assert_eq!(
            bpe.encode_pieces(&[b"abca", b"", b"aaa", long_piece], 20)?,
            [7, 5, 0, 7, 7, 7, 6, 5, 0]
        );
        Ok(())
    }
}
