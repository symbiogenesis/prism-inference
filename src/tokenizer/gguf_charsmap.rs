//! Checked `SentencePiece` XCDA normalizer used by the GGUF unigram tokenizer.

use crate::{Result, invalid};

#[derive(Debug)]
pub(super) struct CharsMap {
    nodes: Vec<u32>,
    replacements: Vec<u8>,
}

impl CharsMap {
    pub(super) fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() > 64 * 1024 * 1024 {
            return Err(invalid("tokenizer charsmap exceeds byte bound"));
        }
        let header = bytes
            .first_chunk::<4>()
            .ok_or_else(|| invalid("tokenizer charsmap header truncated"))?;
        let size = usize::try_from(u32::from_le_bytes(*header))?;
        let end = size
            .checked_add(4)
            .ok_or_else(|| invalid("tokenizer charsmap offset overflow"))?;
        if size == 0 || size % 4 != 0 || end >= bytes.len() {
            return Err(invalid("tokenizer charsmap node extent invalid"));
        }
        let nodes = bytes[4..end]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|value| u32::from_le_bytes(*value))
            .collect();
        let replacements = bytes[end..].to_vec();
        std::str::from_utf8(&replacements)?;
        let result = Self {
            nodes,
            replacements,
        };
        result.node(usize::try_from(Self::base(result.node(0)?))?)?;
        Ok(result)
    }

    const fn base(node: u32) -> u32 {
        (node >> 10) << ((node & (1 << 9)) >> 6)
    }

    fn node(&self, index: usize) -> Result<u32> {
        self.nodes
            .get(index)
            .copied()
            .ok_or_else(|| invalid("tokenizer charsmap node offset exceeds extent"))
    }

    pub(super) fn normalize(&self, input: &str) -> Result<String> {
        let mut output = String::new();
        let mut offset = 0;
        while offset < input.len() {
            let mut index = Self::base(self.node(0)?);
            let mut longest = None;
            for (position, &byte) in input.as_bytes()[offset..].iter().enumerate() {
                if byte == 0 {
                    break;
                }
                index ^= u32::from(byte);
                let node = self.node(usize::try_from(index)?)?;
                if node & ((1 << 31) | 0xff) != u32::from(byte) {
                    break;
                }
                index ^= Self::base(node);
                if node & (1 << 8) != 0 {
                    let replacement = self.node(usize::try_from(index)?)? & 0x7fff_ffff;
                    longest = Some((position + 1, usize::try_from(replacement)?));
                }
            }
            if let Some((consumed, replacement)) = longest {
                let tail = self
                    .replacements
                    .get(replacement..)
                    .ok_or_else(|| invalid("tokenizer charsmap replacement offset invalid"))?;
                let length = tail
                    .iter()
                    .position(|&value| value == 0)
                    .ok_or_else(|| invalid("tokenizer charsmap replacement is unterminated"))?;
                output.push_str(std::str::from_utf8(&tail[..length])?);
                offset += consumed;
            } else {
                let value = input[offset..]
                    .chars()
                    .next()
                    .ok_or_else(|| invalid("tokenizer charsmap scalar missing"))?;
                output.push(value);
                offset += value.len_utf8();
            }
            if output.len() > 1024 * 1024 {
                return Err(invalid("tokenizer normalized text exceeds byte bound"));
            }
        }
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn malformed_charsmap_offsets_fail_without_indexing_panics() -> Result<()> {
        assert!(CharsMap::from_bytes(&[]).is_err());
        assert!(CharsMap::from_bytes(&[0, 0, 0, 0, 0]).is_err());
        assert!(CharsMap::from_bytes(&[4, 0, 0, 0, 0, 4, 0, 0, 0]).is_err());
        let map = CharsMap::from_bytes(&[4, 0, 0, 0, 0, 0, 0, 0, 0])?;
        assert!(map.normalize("x").is_err());
        Ok(())
    }
}
