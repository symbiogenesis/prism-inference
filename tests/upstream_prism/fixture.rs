//! Bounded, strict binary input and bridge response parsing.

use std::io::{Cursor, Read};

use super::{Result, invalid};

pub(super) struct Fixture<'a> {
    bytes: Cursor<&'a [u8]>,
}

impl<'a> Fixture<'a> {
    pub(super) fn new(bytes: &'a [u8], magic: [u8; 8]) -> Result<Self> {
        let mut fixture = Self {
            bytes: Cursor::new(bytes),
        };
        if fixture.block(8)? != magic {
            return Err(invalid("upstream binary magic/version mismatch"));
        }
        Ok(fixture)
    }

    pub(super) fn block(&mut self, length: usize) -> Result<Vec<u8>> {
        let remaining = u64::try_from(self.bytes.get_ref().len())? - self.bytes.position();
        if u64::try_from(length)? > remaining {
            return Err(invalid("truncated upstream binary payload"));
        }
        let mut value = vec![0; length];
        self.bytes.read_exact(&mut value)?;
        Ok(value)
    }

    pub(super) fn u32(&mut self) -> Result<u32> {
        let mut bytes = [0; 4];
        self.bytes.read_exact(&mut bytes)?;
        Ok(u32::from_le_bytes(bytes))
    }

    pub(super) fn floats(&mut self, count: usize) -> Result<Vec<f32>> {
        let bytes = self.block(
            count
                .checked_mul(4)
                .ok_or_else(|| invalid("float size overflow"))?,
        )?;
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&bytes| {
                let value = f32::from_le_bytes(bytes);
                if value.is_finite() {
                    Ok(value)
                } else {
                    Err(invalid("nonfinite upstream binary value"))
                }
            })
            .collect()
    }

    pub(super) fn finish(&self) -> Result<()> {
        if self.bytes.position() != u64::try_from(self.bytes.get_ref().len())? {
            return Err(invalid("trailing bytes in upstream binary payload"));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_magic_truncation_nonfinite_and_trailing_bytes() -> Result<()> {
        assert!(Fixture::new(b"BSPQ2V1\n", *b"BSPQ2I1\n").is_err());
        let mut fixture = Fixture::new(b"BSPQ2I1\n", *b"BSPQ2I1\n")?;
        assert!(fixture.block(usize::MAX).is_err());
        let mut bytes = b"BSPQ2I1\n".to_vec();
        bytes.extend(f32::NAN.to_le_bytes());
        let mut fixture = Fixture::new(&bytes, *b"BSPQ2I1\n")?;
        assert!(fixture.finish().is_err());
        assert!(fixture.floats(1).is_err());
        Ok(())
    }
}
