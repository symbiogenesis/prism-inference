//! Input-only, safe Rust mapping of official llama2.c F32 weights to GGUF.
//! The seven-int header and sequential arrays follow karpathy/llama2.c and
//! pinned examples/convert-llama2c-to-ggml. No quantizer/model API is called.
use super::engine::Engine;
use super::{Result, invalid};
use sha2::{Digest, Sha256};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
const RAW_LENGTH: u64 = 167_020_572;
const RAW_SHA: &str = "9f65a1000e17d0bc167dd6332e0ce5119a0222a3d920cead5bce413bfab2ee7b";
const VOCAB_SHA: &str = "16c3724582d59aa8bf84711894e833f916ee46a31d80e21312759c48bf8d0e69";
fn require(v: bool, message: &str) -> Result<()> {
    if v { Ok(()) } else { Err(invalid(message)) }
}
fn digest(path: &Path) -> Result<String> {
    let mut input = File::open(path)?;
    let mut hash = Sha256::new();
    let mut block = [0_u8; 8192];
    loop {
        let n = input.read(&mut block)?;
        if n == 0 {
            break;
        }
        hash.update(&block[..n]);
    }
    Ok(hash
        .finalize()
        .iter()
        .flat_map(|b| [b >> 4, b & 15])
        .map(|n| char::from(b"0123456789abcdef"[usize::from(n)]))
        .collect())
}
struct Cursor<'a> {
    bytes: &'a [u8],
    position: usize,
}
impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self
            .position
            .checked_add(n)
            .ok_or_else(|| invalid("GGUF cursor overflow"))?;
        let value = self
            .bytes
            .get(self.position..end)
            .ok_or_else(|| invalid("GGUF input truncated"))?;
        self.position = end;
        Ok(value)
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into()?))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into()?))
    }
    fn string(&mut self) -> Result<&'a [u8]> {
        let n = usize::try_from(self.u64()?)?;
        require(n <= 1_048_576, "GGUF input string exceeds bound")?;
        self.take(n)
    }
    fn skip(&mut self, kind: u32, depth: u32) -> Result<()> {
        require(depth < 4, "GGUF array nesting exceeds bound")?;
        let count = match kind {
            0 | 1 | 7 => 1,
            2 | 3 => 2,
            4..=6 => 4,
            10..=12 => 8,
            8 => {
                self.string()?;
                return Ok(());
            }
            9 => {
                let element = self.u32()?;
                let n = self.u64()?;
                require(n <= 1_048_576, "GGUF input array exceeds bound")?;
                for _ in 0..n {
                    self.skip(element, depth + 1)?;
                }
                return Ok(());
            }
            _ => return Err(invalid("GGUF value type unknown")),
        };
        self.take(count)?;
        Ok(())
    }
}
struct Entry {
    name: String,
    kind: u32,
    data: Vec<u8>,
}
fn string_data(text: &str) -> Result<Vec<u8>> {
    let mut bytes = u64::try_from(text.len())?.to_le_bytes().to_vec();
    bytes.extend_from_slice(text.as_bytes());
    Ok(bytes)
}
fn string(name: &str, value: &str) -> Result<Entry> {
    Ok(Entry {
        name: name.into(),
        kind: 8,
        data: string_data(value)?,
    })
}
fn integer(name: &str, value: u32) -> Entry {
    Entry {
        name: name.into(),
        kind: 4,
        data: value.to_le_bytes().to_vec(),
    }
}
fn vocabulary(path: &Path, count: u64) -> Result<Vec<Entry>> {
    require(
        fs::metadata(path)?.len() == 723_869 && digest(path)? == VOCAB_SHA,
        "pinned input vocabulary identity differs",
    )?;
    let bytes = fs::read(path)?;
    let mut cursor = Cursor {
        bytes: &bytes,
        position: 0,
    };
    require(
        cursor.take(4)? == b"GGUF" && cursor.u32()? == 3,
        "input vocabulary GGUF version differs",
    )?;
    let _tensors = cursor.u64()?;
    let keys = cursor.u64()?;
    require(keys <= 1024, "input vocabulary key count exceeds bound")?;
    let mut entries = Vec::new();
    let mut actual_count = None;
    for _ in 0..keys {
        let name = std::str::from_utf8(cursor.string()?)?.to_owned();
        let kind = cursor.u32()?;
        let start = cursor.position;
        cursor.skip(kind, 0)?;
        if name.starts_with("tokenizer.") {
            let data = bytes[start..cursor.position].to_vec();
            if name == "tokenizer.ggml.tokens" {
                require(kind == 9, "vocabulary tokens are not array")?;
                let mut value = Cursor {
                    bytes: &data,
                    position: 0,
                };
                require(value.u32()? == 8, "vocabulary tokens are not strings")?;
                actual_count = Some(value.u64()?);
            }
            entries.push(Entry { name, kind, data });
        }
    }
    require(
        actual_count == Some(count),
        "input vocabulary count differs from official weights",
    )?;
    Ok(entries)
}
struct Tensor {
    name: String,
    dimensions: Vec<u64>,
    source: u64,
    bytes: u64,
}
fn section(cursor: &mut u64, elements: u64) -> Result<u64> {
    let start = *cursor;
    let bytes = elements
        .checked_mul(4)
        .ok_or_else(|| invalid("weight section byte overflow"))?;
    *cursor = cursor
        .checked_add(bytes)
        .ok_or_else(|| invalid("weight section offset overflow"))?;
    Ok(start)
}
fn tensor(name: &str, dimensions: &[u64], source: u64) -> Result<Tensor> {
    let elements = dimensions.iter().try_fold(1_u64, |a, b| {
        a.checked_mul(*b)
            .ok_or_else(|| invalid("input tensor extent overflow"))
    })?;
    require(
        elements > 0 && elements % 256 == 0,
        "official default tensor violates quantizer block preconditions",
    )?;
    Ok(Tensor {
        name: name.into(),
        dimensions: dimensions.to_vec(),
        source,
        bytes: elements
            .checked_mul(4)
            .ok_or_else(|| invalid("tensor bytes overflow"))?,
    })
}
fn map_weights(path: &Path) -> Result<(Vec<Entry>, Vec<Tensor>)> {
    require(
        fs::metadata(path)?.len() == RAW_LENGTH && digest(path)? == RAW_SHA,
        "official42M raw input identity differs",
    )?;
    let mut input = File::open(path)?;
    let mut header = [0_u8; 28];
    input.read_exact(&mut header)?;
    let values = header
        .as_chunks::<4>()
        .0
        .iter()
        .map(|v| i32::from_le_bytes(*v))
        .collect::<Vec<_>>();
    require(
        values == [512, 1376, 8, 8, 8, 32000, 1024],
        "official42M header geometry differs",
    )?;
    let dimension = 512_u64;
    let hidden = 1376_u64;
    let layers = 8_u64;
    let heads = 8_u64;
    let vocab = 32000_u64;
    let context = 1024_u64;
    let mut offset = 28_u64;
    let embedding = section(&mut offset, vocab * dimension)?;
    let attention_norm = section(&mut offset, layers * dimension)?;
    let query = section(&mut offset, layers * dimension * dimension)?;
    let key = section(&mut offset, layers * dimension * dimension)?;
    let value = section(&mut offset, layers * dimension * dimension)?;
    let attention_output = section(&mut offset, layers * dimension * dimension)?;
    let ffn_norm = section(&mut offset, layers * dimension)?;
    let gate = section(&mut offset, layers * hidden * dimension)?;
    let down = section(&mut offset, layers * hidden * dimension)?;
    let up = section(&mut offset, layers * hidden * dimension)?;
    let final_norm = section(&mut offset, dimension)?;
    section(&mut offset, context * (dimension / heads))?;
    require(
        offset == RAW_LENGTH,
        "official input array extents do not exhaust file",
    )?;
    let mut tensors = vec![
        tensor("token_embd.weight", &[dimension, vocab], embedding)?,
        tensor("output_norm.weight", &[dimension], final_norm)?,
        tensor("output.weight", &[dimension, vocab], embedding)?,
    ];
    for layer in 0..layers {
        let dense = layer * dimension * dimension * 4;
        let ff = layer * hidden * dimension * 4;
        let norm = layer * dimension * 4;
        for (name, dims, start) in [
            ("attn_q.weight", vec![dimension, dimension], query + dense),
            ("attn_k.weight", vec![dimension, dimension], key + dense),
            ("attn_v.weight", vec![dimension, dimension], value + dense),
            (
                "attn_output.weight",
                vec![dimension, dimension],
                attention_output + dense,
            ),
            ("attn_norm.weight", vec![dimension], attention_norm + norm),
            ("ffn_gate.weight", vec![dimension, hidden], gate + ff),
            ("ffn_down.weight", vec![hidden, dimension], down + ff),
            ("ffn_up.weight", vec![dimension, hidden], up + ff),
            ("ffn_norm.weight", vec![dimension], ffn_norm + norm),
        ] {
            tensors.push(tensor(&format!("blk.{layer}.{name}"), &dims, start)?);
        }
    }
    let entries = vec![
        string("general.architecture", "llama")?,
        string("general.name", "official TinyStories42M F32")?,
        integer("general.file_type", 0),
        integer("llama.context_length", 1024),
        integer("llama.embedding_length", 512),
        integer("llama.feed_forward_length", 1376),
        integer("llama.attention.head_count", 8),
        integer("llama.attention.head_count_kv", 8),
        integer("llama.block_count", 8),
        integer("llama.rope.dimension_count", 64),
        Entry {
            name: "llama.attention.layer_norm_rms_epsilon".into(),
            kind: 6,
            data: 1e-5_f32.to_le_bytes().to_vec(),
        },
    ];
    Ok((entries, tensors))
}
fn write(path: &Path, raw: &Path, entries: &[Entry], tensors: &[Tensor]) -> Result<()> {
    let mut header = b"GGUF".to_vec();
    header.extend_from_slice(&3_u32.to_le_bytes());
    header.extend_from_slice(&u64::try_from(tensors.len())?.to_le_bytes());
    header.extend_from_slice(&u64::try_from(entries.len())?.to_le_bytes());
    for entry in entries {
        header.extend(string_data(&entry.name)?);
        header.extend_from_slice(&entry.kind.to_le_bytes());
        header.extend_from_slice(&entry.data);
    }
    let mut offset = 0_u64;
    for tensor in tensors {
        header.extend(string_data(&tensor.name)?);
        header.extend_from_slice(&u32::try_from(tensor.dimensions.len())?.to_le_bytes());
        for dimension in &tensor.dimensions {
            header.extend_from_slice(&dimension.to_le_bytes());
        }
        header.extend_from_slice(&0_u32.to_le_bytes());
        header.extend_from_slice(&offset.to_le_bytes());
        offset = offset
            .checked_add(tensor.bytes)
            .ok_or_else(|| invalid("GGUF output extent overflow"))?;
    }
    header.resize(
        header
            .len()
            .checked_add(31)
            .ok_or_else(|| invalid("GGUF padding overflow"))?
            & !31,
        0,
    );
    let mut output = OpenOptions::new().create_new(true).write(true).open(path)?;
    output.write_all(&header)?;
    let mut source = File::open(raw)?;
    for tensor in tensors {
        source.seek(SeekFrom::Start(tensor.source))?;
        let copied = std::io::copy(
            &mut Read::by_ref(&mut source).take(tensor.bytes),
            &mut output,
        )?;
        require(
            copied == tensor.bytes,
            "raw input truncated while copying weights",
        )?;
    }
    output.flush()?;
    Ok(())
}
/// Creates an independent, float-only fixture with every original tensor intact.
pub(super) fn prepare(engine: &Engine, directory: &Path) -> Result<PathBuf> {
    let raw = engine.asset_path("tinyllamas/stories42M.bin")?;
    let vocab = engine.source_path("models/ggml-vocab-llama-spm.gguf")?;
    prepare_paths(&raw, &vocab, directory)
}
pub(super) fn prepare_paths(raw: &Path, vocab: &Path, directory: &Path) -> Result<PathBuf> {
    let (mut entries, tensors) = map_weights(raw)?;
    entries.extend(vocabulary(vocab, 32000)?);
    let output = directory.join("official-stories42M-f32.gguf");
    write(&output, raw, &entries, &tensors)?;
    require(
        fs::metadata(&output)?.len() == 233_022_656
            && digest(&output)?
                == "05429b48a224122f59d25e136758d240fd59fce884c4da3465bae9c64d210269",
        "converted independent input identity differs",
    )?;
    Ok(output)
}
