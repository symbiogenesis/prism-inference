//! Streaming 64-bit block hasher from pinned Prism's MIT-licensed Jinja runtime.
//!
//! This is a hash-table helper, with no cryptographic security property.

const SEED: u64 = 0xcbf2_9ce4_8422_2325;
const PRIME: u64 = 0x0100_0000_01b3;

/// Incremental, little-endian block hash for the selected 64-bit targets.
#[derive(Clone, Debug)]
pub struct Hasher {
    state: u64,
    buffer: [u8; 8],
    index: usize,
}

impl Default for Hasher {
    fn default() -> Self {
        Self::new()
    }
}

impl Hasher {
    /// Creates an empty stream with the pinned initial state.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            state: SEED,
            buffer: [0; 8],
            index: 0,
        }
    }

    const fn block(&mut self, block: [u8; 8]) {
        self.state = (self.state ^ u64::from_le_bytes(block)).wrapping_mul(PRIME);
    }

    /// Adds bytes without introducing a boundary between adjacent updates.
    /// Empty updates never change state, including after a digest.
    pub fn update(&mut self, mut bytes: &[u8]) {
        if self.index > 0 {
            let count = (8 - self.index).min(bytes.len());
            self.buffer[self.index..self.index + count].copy_from_slice(&bytes[..count]);
            self.index += count;
            bytes = &bytes[count..];
            if self.index == 8 {
                self.block(self.buffer);
                self.index = 0;
            }
        }
        let (blocks, remainder) = bytes.as_chunks::<8>();
        for block in blocks {
            self.block(*block);
        }
        if !remainder.is_empty() {
            self.buffer[..remainder.len()].copy_from_slice(remainder);
            self.index = remainder.len();
        }
    }

    /// Zero-pads the pending block and returns the resulting state.
    ///
    /// Finalization mutates this object. A subsequent digest is stable; later
    /// updates begin a new block using the already-finalized state, matching
    /// the pinned source's intermediate-digest behavior.
    pub fn digest(&mut self) -> u64 {
        if self.index > 0 {
            self.buffer[self.index..].fill(0);
            self.block(self.buffer);
            self.index = 0;
        }
        self.state
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intermediate_digest_matches_block_arithmetic() {
        let mut traced = Hasher::new();
        traced.update(b"a");
        let first = (SEED ^ u64::from(b'a')).wrapping_mul(PRIME);
        assert_eq!(traced.digest(), first);
        traced.update(&[]);
        assert_eq!(traced.digest(), first);
        traced.update(b"b");
        assert_eq!(
            traced.digest(),
            (first ^ u64::from(b'b')).wrapping_mul(PRIME)
        );
    }
}
