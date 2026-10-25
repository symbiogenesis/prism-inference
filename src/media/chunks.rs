//! Owned multimodal chunks and the pinned little-endian MTMD v1 cache format.
//!
//! Ported from the MIT-licensed `tools/mtmd/mtmd.cpp` at the pinned Prism
//! revision. Serialized image frames are placeholders, as in the production
//! format; pixel buffers are deliberately not part of a cache record.

use crate::{Result, invalid};

const LIMIT: usize = 1024 * 1024;

/// The decoder position layout of an image chunk.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PositionLayout {
    /// Advance each coordinate for every token.
    Sequential,
    /// Share a temporal position with separate image X/Y coordinates.
    MultiRope,
    /// Include begin/end and row-newline positions for `HunyuanVL`.
    Hunyuan,
}

/// Metadata retained for one preprocessed frame after cache loading.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Frame {
    /// Whether a separator follows this view.
    pub view_separator: bool,
    /// Whether a newline follows this view.
    pub newline: bool,
    /// Original frame width.
    pub width: i32,
    /// Original frame height.
    pub height: i32,
}

/// Image grid, temporal layout and placeholder frame metadata.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Image {
    /// Number of horizontal token positions.
    pub width: u32,
    /// Number of vertical token positions.
    pub height: u32,
    /// Decoder coordinate layout.
    pub layout: PositionLayout,
    /// Image ordinal used by the Hunyuan layout.
    pub image_index: u32,
    /// Temporal frame merge factor.
    pub temporal_merge: u32,
    /// User-defined KV cache identity.
    pub id: String,
    /// Independent placeholder frame descriptors.
    pub frames: Vec<Frame>,
}

impl Image {
    /// Number of decoder tokens, including layout-specific marker positions.
    ///
    /// # Errors
    /// Rejects zero temporal merge, arithmetic overflow or excessive token count.
    pub fn token_count(&self) -> Result<usize> {
        if self.temporal_merge == 0 {
            return Err(invalid("image temporal merge must be nonzero"));
        }
        let count = if self.layout == PositionLayout::Hunyuan {
            (u64::from(self.width) + 1) * u64::from(self.height) + 2
        } else {
            let frames = u64::try_from(self.frames.len())?.div_ceil(u64::from(self.temporal_merge));
            u64::from(self.width)
                .checked_mul(u64::from(self.height))
                .and_then(|area| area.checked_mul(frames))
                .ok_or_else(|| invalid("image token count overflow"))?
        };
        if count > 1_048_576 {
            return Err(invalid("image token count exceeds bound"));
        }
        Ok(usize::try_from(count)?)
    }

    /// Four decoder coordinates for one image token.
    ///
    /// # Errors
    /// Rejects an out-of-range index, invalid grid, or overflowing position.
    pub fn position(&self, start: u32, index: usize) -> Result<[u64; 4]> {
        let count = self.token_count()?;
        if index >= count || self.width == 0 {
            return Err(invalid("image position outside token grid"));
        }
        let sequential = u64::from(start) + u64::try_from(index)?;
        Ok(match self.layout {
            PositionLayout::Sequential => [sequential; 4],
            PositionLayout::MultiRope => [
                u64::from(start),
                u64::from(start) + u64::try_from(index)? % u64::from(self.width),
                u64::from(start) + u64::try_from(index)? / u64::from(self.width),
                0,
            ],
            PositionLayout::Hunyuan if index == 0 || index == count - 1 => [sequential; 4],
            PositionLayout::Hunyuan => {
                let offset = u64::try_from(index - 1)?;
                [
                    sequential,
                    offset / (u64::from(self.width) + 1),
                    offset % (u64::from(self.width) + 1),
                    u64::from(self.image_index),
                ]
            }
        })
    }
}

/// Audio token count, user identity and placeholder frame metadata.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Audio {
    /// Number of decoder tokens/positions.
    pub tokens: u32,
    /// User-defined KV cache identity.
    pub id: String,
    /// Independent placeholder frame descriptors.
    pub frames: Vec<Frame>,
}

/// Independently owned text, image or audio input chunk.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Chunk {
    /// Token IDs; no multimodal payload is present.
    Text(Vec<i32>),
    /// Image token-grid metadata.
    Image(Image),
    /// Audio token metadata.
    Audio(Audio),
}

impl Chunk {
    /// Number of decoder tokens in the chunk.
    ///
    /// # Errors
    /// Rejects an invalid image layout or excessive token count.
    pub fn token_count(&self) -> Result<usize> {
        match self {
            Self::Text(tokens) => Ok(tokens.len()),
            Self::Image(image) => image.token_count(),
            Self::Audio(audio) => Ok(usize::try_from(audio.tokens)?),
        }
    }

    /// Sequence position span occupied by the chunk.
    ///
    /// # Errors
    /// Rejects malformed image metadata or overflowing counts.
    pub fn position_count(&self) -> Result<usize> {
        match self {
            Self::Image(image) if image.layout == PositionLayout::MultiRope => {
                Ok(usize::try_from(image.width.max(image.height))?)
            }
            _ => self.token_count(),
        }
    }

    /// User KV cache identity for media chunks; text has none.
    #[must_use]
    pub fn id(&self) -> Option<&str> {
        match self {
            Self::Text(_) => None,
            Self::Image(image) => Some(&image.id),
            Self::Audio(audio) => Some(&audio.id),
        }
    }

    /// Serializes independently owned metadata using the upstream v1 layout.
    ///
    /// # Errors
    /// Rejects invalid metadata or a record larger than 1 MiB.
    pub fn save(&self) -> Result<Vec<u8>> {
        self.token_count()?;
        let mut bytes = 1_u64.to_le_bytes().to_vec();
        let kind: u32 = match self {
            Self::Text(_) => 0,
            Self::Image(_) => 1,
            Self::Audio(_) => 2,
        };
        bytes.extend(kind.to_le_bytes());
        let text = if let Self::Text(tokens) = self {
            tokens.as_slice()
        } else {
            &[]
        };
        if text.len() > LIMIT / 4 {
            return Err(invalid("text chunk exceeds cache bound"));
        }
        bytes.extend(u64::try_from(text.len())?.to_le_bytes());
        for token in text {
            bytes.extend(token.to_le_bytes());
        }
        bytes.push(u8::from(matches!(self, Self::Image(_))));
        if let Self::Image(image) = self {
            for value in [
                image.width,
                image.height,
                match image.layout {
                    PositionLayout::Sequential => 0,
                    PositionLayout::MultiRope => 1,
                    PositionLayout::Hunyuan => 2,
                },
                image.image_index,
                image.temporal_merge,
            ] {
                bytes.extend(value.to_le_bytes());
            }
            string(&mut bytes, &image.id)?;
            frames(&mut bytes, &image.frames, false)?;
        }
        bytes.push(u8::from(matches!(self, Self::Audio(_))));
        if let Self::Audio(audio) = self {
            bytes.extend(audio.tokens.to_le_bytes());
            string(&mut bytes, &audio.id)?;
            frames(&mut bytes, &audio.frames, true)?;
        }
        if bytes.len() > LIMIT {
            return Err(invalid("chunk cache record exceeds1MiB"));
        }
        Ok(bytes)
    }

    /// Loads a bounded upstream v1 record, retaining placeholder frame metadata.
    ///
    /// As in upstream, bytes following the complete record are ignored.
    ///
    /// # Errors
    /// Rejects truncated, oversized, mismatched or invalid-layout records.
    pub fn load(bytes: &[u8]) -> Result<Self> {
        if bytes.len() > LIMIT {
            return Err(invalid("chunk cache record exceeds1MiB"));
        }
        let mut input = Reader { bytes, position: 0 };
        if input.u64()? != 1 {
            return Err(invalid("chunk cache version mismatch"));
        }
        let kind = input.u32()?;
        let count = usize::try_from(input.u64()?)?;
        if count > input.remaining() / 4 {
            return Err(invalid("text token extent exceeds chunk cache"));
        }
        let mut text = Vec::with_capacity(count);
        for _ in 0..count {
            text.push(i32::from_le_bytes(input.take()?));
        }
        let image = if input.boolean()? {
            Some(Image {
                width: input.u32()?,
                height: input.u32()?,
                layout: match input.u32()? {
                    0 => PositionLayout::Sequential,
                    1 => PositionLayout::MultiRope,
                    2 => PositionLayout::Hunyuan,
                    _ => return Err(invalid("invalid image position layout")),
                },
                image_index: input.u32()?,
                temporal_merge: input.u32()?,
                id: input.string()?,
                frames: input.frames()?,
            })
        } else {
            None
        };
        let audio = if input.boolean()? {
            Some(Audio {
                tokens: input.u32()?,
                id: input.string()?,
                frames: input.frames()?,
            })
        } else {
            None
        };
        let chunk = match kind {
            0 => Self::Text(text),
            1 => Self::Image(image.ok_or_else(|| invalid("image chunk has no image payload"))?),
            2 => Self::Audio(audio.ok_or_else(|| invalid("audio chunk has no audio payload"))?),
            _ => return Err(invalid("unknown chunk type")),
        };
        chunk.token_count()?;
        Ok(chunk)
    }
}

fn string(bytes: &mut Vec<u8>, value: &str) -> Result<()> {
    if value.len() > LIMIT {
        return Err(invalid("chunk cache ID exceeds bound"));
    }
    bytes.extend(u64::try_from(value.len())?.to_le_bytes());
    bytes.extend(value.as_bytes());
    Ok(())
}

fn frames(bytes: &mut Vec<u8>, values: &[Frame], audio: bool) -> Result<()> {
    if values.len() > LIMIT / 10 {
        return Err(invalid("chunk frame metadata exceeds cache bound"));
    }
    bytes.push(u8::from(audio));
    bytes.extend(u64::try_from(values.len())?.to_le_bytes());
    for frame in values {
        bytes.extend([u8::from(frame.view_separator), u8::from(frame.newline)]);
        bytes.extend(frame.width.to_le_bytes());
        bytes.extend(frame.height.to_le_bytes());
    }
    Ok(())
}

struct Reader<'a> {
    bytes: &'a [u8],
    position: usize,
}
impl Reader<'_> {
    const fn remaining(&self) -> usize {
        self.bytes.len() - self.position
    }
    fn take<const N: usize>(&mut self) -> Result<[u8; N]> {
        let end = self
            .position
            .checked_add(N)
            .ok_or_else(|| invalid("chunk cursor overflow"))?;
        let value = self
            .bytes
            .get(self.position..end)
            .ok_or_else(|| invalid("truncated chunk cache"))?
            .try_into()?;
        self.position = end;
        Ok(value)
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take()?))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take()?))
    }
    fn boolean(&mut self) -> Result<bool> {
        Ok(self.take::<1>()?[0] != 0)
    }
    fn string(&mut self) -> Result<String> {
        let length = usize::try_from(self.u64()?)?;
        if length > self.remaining() {
            return Err(invalid("chunk ID exceeds record extent"));
        }
        let text =
            std::str::from_utf8(&self.bytes[self.position..self.position + length])?.to_owned();
        self.position += length;
        Ok(text)
    }
    fn frames(&mut self) -> Result<Vec<Frame>> {
        self.boolean()?;
        let count = usize::try_from(self.u64()?)?;
        if count > self.remaining() / 10 {
            return Err(invalid("chunk frame count exceeds record extent"));
        }
        let mut frames = Vec::with_capacity(count);
        for _ in 0..count {
            frames.push(Frame {
                view_separator: self.boolean()?,
                newline: self.boolean()?,
                width: i32::from_le_bytes(self.take()?),
                height: i32::from_le_bytes(self.take()?),
            });
        }
        Ok(frames)
    }
}

/// Public fixture factory preserved from the original production MTMD API.
#[must_use]
pub fn test_inputs() -> Vec<Chunk> {
    vec![
        Chunk::Text(vec![1, 2, 3, 4, 5]),
        Chunk::Image(Image {
            width: 4,
            height: 4,
            layout: PositionLayout::Sequential,
            image_index: 0,
            temporal_merge: 1,
            id: "image_1".into(),
            frames: vec![Frame::default(); 16],
        }),
    ]
}
