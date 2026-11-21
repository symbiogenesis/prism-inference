//! Raw bit-preserving token candidate buffer storage and codecs for `LLGuidance` reference sessions.
//!
//! Stores `(id: i32, logit_bits: u32, probability_bits: u32)` strictly as raw 32-bit patterns so
//! negative zeros (`-0.0`) and quiet/signaling NaN payloads are never canonicalized or lost.

use std::fs::{self, File};
use std::io::Read;
use std::path::Path;

use serde_json::{Value, json};

use super::super::llama_grammar::{exact_fields, field, uint};
use super::super::{Result, invalid};

const MAX_CANDIDATE_FILE_BYTES: u64 = 64 * 1024 * 1024;
const MAX_CANDIDATES: usize = 1_048_576;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct CandidateEntry {
    pub(super) id: i32,
    pub(super) logit_bits: u32,
    pub(super) prob_bits: u32,
}

impl CandidateEntry {
    pub(super) fn to_json(self) -> Value {
        json!({
            "Id": self.id,
            "LogitBits": self.logit_bits,
            "ProbabilityBits": self.prob_bits,
        })
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum CopyEncoding {
    Candidate,
    IdLogit,
    Logits,
}

impl CopyEncoding {
    pub(super) fn parse(name: &str) -> Result<Self> {
        match name {
            "candidate-le" => Ok(Self::Candidate),
            "id-logit-le" => Ok(Self::IdLogit),
            "logits-le" => Ok(Self::Logits),
            _ => Err(invalid(format!(
                "unsupported LLG candidate encoding {name}"
            ))),
        }
    }

    const fn stride(self) -> usize {
        match self {
            Self::Candidate => 12,
            Self::IdLogit => 8,
            Self::Logits => 4,
        }
    }
}

#[derive(Clone)]
pub(super) struct CandidateBuffer {
    pub(super) entries: Vec<CandidateEntry>,
    pub(super) offset: usize,
    pub(super) size: usize,
    pub(super) selected: i64,
    pub(super) sorted: bool,
}

impl CandidateBuffer {
    pub(super) fn from_file(
        path: &Path,
        count: usize,
        selected: i64,
        sorted: bool,
    ) -> Result<Self> {
        if count == 0 || count > MAX_CANDIDATES {
            return Err(invalid("LLG candidate buffer count out of bounds"));
        }
        let bytes = read_bounded_bytes(path, count.saturating_mul(12))?;
        let mut entries = Vec::with_capacity(count);
        for chunk in bytes.as_chunks::<12>().0 {
            entries.push(CandidateEntry {
                id: i32::from_le_bytes(chunk[0..4].try_into()?),
                logit_bits: u32::from_le_bytes(chunk[4..8].try_into()?),
                prob_bits: u32::from_le_bytes(chunk[8..12].try_into()?),
            });
        }
        let buf = Self {
            entries,
            offset: 0,
            size: count,
            selected,
            sorted,
        };
        buf.validate_selected()?;
        Ok(buf)
    }

    fn validate_selected(&self) -> Result<()> {
        if self.selected < -1 {
            return Err(invalid("LLG candidate selected index below -1"));
        }
        if self.selected >= 0 {
            let sel = usize::try_from(self.selected)?;
            if sel >= self.size {
                return Err(invalid("LLG candidate selected index exceeds active slice"));
            }
        }
        Ok(())
    }

    pub(super) fn apply_metadata(&mut self, metadata: &Value) -> Result<()> {
        if metadata.is_null() {
            return Ok(());
        }
        exact_fields(metadata, &["Offset", "Size", "Selected", "Sorted"])?;
        let offset = usize::try_from(uint(metadata, "Offset")?)?;
        let size = usize::try_from(uint(metadata, "Size")?)?;
        let selected = field(metadata, "Selected")?
            .as_i64()
            .ok_or_else(|| invalid("LLG Selected must be integer"))?;
        let sorted = field(metadata, "Sorted")?
            .as_bool()
            .ok_or_else(|| invalid("LLG Sorted must be boolean"))?;
        let end = offset
            .checked_add(size)
            .ok_or_else(|| invalid("LLG offset + size overflow"))?;
        if end > self.entries.len() {
            return Err(invalid("LLG metadata slice exceeds buffer size"));
        }
        self.offset = offset;
        self.size = size;
        self.selected = selected;
        self.sorted = sorted;
        self.validate_selected()?;
        Ok(())
    }

    pub(super) fn load_file(
        &mut self,
        path: &Path,
        encoding: CopyEncoding,
        metadata: &Value,
    ) -> Result<()> {
        let count = self.entries.len();
        let bytes = read_bounded_bytes(path, count.saturating_mul(encoding.stride()))?;
        match encoding {
            CopyEncoding::Candidate => {
                for (dst, chunk) in self.entries.iter_mut().zip(bytes.as_chunks::<12>().0) {
                    dst.id = i32::from_le_bytes(chunk[0..4].try_into()?);
                    dst.logit_bits = u32::from_le_bytes(chunk[4..8].try_into()?);
                    dst.prob_bits = u32::from_le_bytes(chunk[8..12].try_into()?);
                }
            }
            CopyEncoding::IdLogit => {
                for (dst, chunk) in self.entries.iter_mut().zip(bytes.as_chunks::<8>().0) {
                    dst.id = i32::from_le_bytes(chunk[0..4].try_into()?);
                    dst.logit_bits = u32::from_le_bytes(chunk[4..8].try_into()?);
                }
            }
            CopyEncoding::Logits => {
                for (dst, &chunk) in self.entries.iter_mut().zip(bytes.as_chunks::<4>().0) {
                    dst.logit_bits = u32::from_le_bytes(chunk);
                }
            }
        }
        self.apply_metadata(metadata)?;
        Ok(())
    }

    pub(super) fn copy_from(
        &mut self,
        source: &Self,
        encoding: CopyEncoding,
        metadata: &Value,
    ) -> Result<()> {
        if self.entries.len() != source.entries.len() {
            return Err(invalid("LLG candidate source/destination lengths differ"));
        }
        match encoding {
            CopyEncoding::Candidate => {
                self.entries.clone_from(&source.entries);
            }
            CopyEncoding::IdLogit => {
                for (dst, src) in self.entries.iter_mut().zip(&source.entries) {
                    dst.id = src.id;
                    dst.logit_bits = src.logit_bits;
                }
            }
            CopyEncoding::Logits => {
                for (dst, src) in self.entries.iter_mut().zip(&source.entries) {
                    dst.logit_bits = src.logit_bits;
                }
            }
        }
        self.apply_metadata(metadata)?;
        Ok(())
    }

    pub(super) fn patch(&mut self, patches: &[Value]) -> Result<()> {
        for patch in patches {
            exact_fields(patch, &["Index", "Fields"])?;
            let index = usize::try_from(uint(patch, "Index")?)?;
            let entry = self
                .entries
                .get_mut(index)
                .ok_or_else(|| invalid("LLG patch index outside buffer"))?;
            let fields = field(patch, "Fields")?
                .as_object()
                .ok_or_else(|| invalid("LLG patch Fields must be object"))?;
            for (name, val) in fields {
                match name.as_str() {
                    "Id" => {
                        entry.id = i32::try_from(
                            val.as_i64()
                                .ok_or_else(|| invalid("LLG patch Id must be integer"))?,
                        )?;
                    }
                    "LogitBits" => {
                        let bits = val
                            .as_u64()
                            .ok_or_else(|| invalid("LLG patch LogitBits must be unsigned"))?;
                        entry.logit_bits = u32::try_from(bits)?;
                    }
                    "ProbabilityBits" => {
                        let bits = val
                            .as_u64()
                            .ok_or_else(|| invalid("LLG patch ProbabilityBits must be unsigned"))?;
                        entry.prob_bits = u32::try_from(bits)?;
                    }
                    _ => return Err(invalid(format!("unknown LLG patch field {name}"))),
                }
            }
        }
        Ok(())
    }

    pub(super) fn snapshot(&self, probe_indices: &[usize]) -> Result<Value> {
        let mut probes = Vec::with_capacity(probe_indices.len());
        for &index in probe_indices {
            let entry = self
                .entries
                .get(index)
                .ok_or_else(|| invalid("LLG probe index outside backing buffer"))?;
            probes.push(json!({
                "Index": index,
                "Candidate": entry.to_json(),
            }));
        }
        let selected_backing = if self.selected >= 0 {
            let local = usize::try_from(self.selected)?;
            let idx = self
                .offset
                .checked_add(local)
                .ok_or_else(|| invalid("LLG selected backing index overflow"))?;
            let entry = self
                .entries
                .get(idx)
                .ok_or_else(|| invalid("LLG selected backing index outside buffer"))?;
            entry.to_json()
        } else {
            Value::Null
        };
        Ok(json!({
            "BufferSize": self.entries.len(),
            "Size": self.size,
            "Offset": self.offset,
            "Selected": self.selected,
            "Sorted": self.sorted,
            "SelectedBacking": selected_backing,
            "Probes": probes,
        }))
    }
}

fn read_bounded_bytes(path: &Path, expected_len: usize) -> Result<Vec<u8>> {
    if !path.is_absolute() {
        return Err(invalid("LLG candidate file path must be absolute"));
    }
    let meta = fs::metadata(path)?;
    if !meta.is_file() || meta.len() > MAX_CANDIDATE_FILE_BYTES {
        return Err(invalid("LLG candidate file exceeds bound or is not a file"));
    }
    if meta.len() != u64::try_from(expected_len)? {
        return Err(invalid("LLG candidate file byte length mismatch"));
    }
    let file = File::open(path)?;
    let mut bytes = Vec::with_capacity(expected_len);
    file.take(u64::try_from(expected_len)? + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() != expected_len {
        return Err(invalid("LLG candidate file read byte count mismatch"));
    }
    Ok(bytes)
}
