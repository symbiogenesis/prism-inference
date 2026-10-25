// Copyright (c) 2023-2026 The ggml/llama.cpp authors
// SPDX-License-Identifier: MIT
//! The source's independent stride/block traversal and scalar decoding.

use super::super::{Result, invalid};
use super::program::Program;
use super::transport::{MAX_FILE_BYTES, Session};
use serde::Deserialize;
use serde_json::json;
use std::fs::{self, File};
use std::io::{BufWriter, Read, Write};
use std::path::Path;

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub(super) struct Tensor {
    pub(super) id: u32,
    pub(super) aliases: Vec<u32>,
    #[serde(rename = "Type")]
    pub(super) kind: i32,
    #[serde(rename = "NE")]
    pub(super) ne: [i64; 4],
    #[serde(rename = "NB")]
    pub(super) nb: [u64; 4],
    pub(super) op: i32,
    pub(super) flags: i32,
    pub(super) name: String,
    pub(super) view_source: Option<u32>,
    pub(super) view_offset: u64,
    pub(super) sources: Vec<Option<u32>>,
    pub(super) op_params_hex: String,
    #[serde(rename = "NBytes")]
    pub(super) bytes: u64,
    #[serde(rename = "NElements")]
    pub(super) elements: i64,
    #[serde(rename = "NRows")]
    pub(super) rows: i64,
    pub(super) block_size: i64,
    pub(super) type_size: u64,
    pub(super) op_description: String,
}

fn kind(program: &Program, name: &str) -> Result<i32> {
    let value = program
        .constants
        .get(name)
        .and_then(serde_json::Value::as_i64)
        .ok_or_else(|| invalid(format!("source lacks type constant {name}")))?;
    Ok(i32::try_from(value)?)
}

pub(super) fn fp16_to_float(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exponent = u32::from((bits >> 10) & 31);
    let mut fraction = u32::from(bits & 1023);
    if exponent == 0 {
        if fraction == 0 {
            return f32::from_bits(sign);
        }
        let mut exponent = 113_u32;
        while fraction & 1024 == 0 {
            fraction <<= 1;
            exponent -= 1;
        }
        return f32::from_bits(sign | (exponent << 23) | ((fraction & 1023) << 13));
    }
    if exponent == 31 {
        return f32::from_bits(sign | 0x7f80_0000 | (fraction << 13));
    }
    f32::from_bits(sign | ((exponent + 112) << 23) | (fraction << 13))
}

fn read_bytes(session: &Session, path: &Path, expected: u64) -> Result<Vec<u8>> {
    let path = session.checked_file(path)?;
    let metadata = fs::metadata(&path)?;
    if !metadata.is_file() || metadata.len() != expected || expected > MAX_FILE_BYTES {
        return Err(invalid("backend raw tensor byte extent differs"));
    }
    let mut bytes = Vec::new();
    File::open(path)?
        .take(expected + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() as u64 != expected {
        return Err(invalid("backend raw tensor changed during bounded read"));
    }
    Ok(bytes)
}

fn element<const N: usize>(raw: &[u8], offset: usize) -> Result<[u8; N]> {
    let end = offset
        .checked_add(N)
        .ok_or_else(|| invalid("backend scalar offset overflow"))?;
    raw.get(offset..end)
        .ok_or_else(|| invalid("backend scalar exceeds raw tensor extent"))?
        .try_into()
        .map_err(|_| invalid("backend scalar byte width differs"))
}

impl Tensor {
    fn dimensions(&self) -> Result<([usize; 4], usize)> {
        let mut ne = [0_usize; 4];
        let mut elements = 1_usize;
        for (destination, &dimension) in ne.iter_mut().zip(&self.ne) {
            *destination = usize::try_from(dimension)?;
            elements = elements
                .checked_mul(*destination)
                .ok_or_else(|| invalid("backend element extent overflow"))?;
        }
        let rows = ne[1..].iter().try_fold(1_usize, |count, dimension| {
            count
                .checked_mul(*dimension)
                .ok_or_else(|| invalid("backend row extent overflow"))
        })?;
        if i64::try_from(elements)? != self.elements
            || i64::try_from(rows)? != self.rows
            || self.block_size <= 0
            || self.type_size == 0
        {
            return Err(invalid("backend tensor metadata extent differs"));
        }
        Ok((ne, elements))
    }

    fn offsets(&self, mut visit: impl FnMut(usize) -> Result<()>) -> Result<()> {
        let (ne, _) = self.dimensions()?;
        let block = usize::try_from(self.block_size)?;
        if !ne[0].is_multiple_of(block) {
            return Err(invalid("backend leading dimension is not block aligned"));
        }
        // This order and i0/block addressing are the source's tensor_to_float reference.
        for i3 in 0..ne[3] {
            for i2 in 0..ne[2] {
                for i1 in 0..ne[1] {
                    for i0 in (0..ne[0]).step_by(block) {
                        let mut offset = 0_u64;
                        for (index, stride) in [
                            (i3, self.nb[3]),
                            (i2, self.nb[2]),
                            (i1, self.nb[1]),
                            (i0 / block, self.nb[0]),
                        ] {
                            offset = offset
                                .checked_add(
                                    u64::try_from(index)?.checked_mul(stride).ok_or_else(|| {
                                        invalid("backend stride product overflow")
                                    })?,
                                )
                                .ok_or_else(|| invalid("backend stride offset overflow"))?;
                        }
                        visit(usize::try_from(offset)?)?;
                    }
                }
            }
        }
        Ok(())
    }
}

/// Production quantized block conversion follows a Rust-owned, source-faithful block ordering.
pub(super) fn tensor_to_float(
    session: &mut Session,
    program: &Program,
    tensor: &Tensor,
    path: &Path,
    artifact: &str,
) -> Result<Vec<f32>> {
    let raw = read_bytes(session, path, tensor.bytes)?;
    let (_, elements) = tensor.dimensions()?;
    let mut output = Vec::new();
    output.try_reserve_exact(elements)?;
    let f32_type = kind(program, "GGML_TYPE_F32")?;
    let f16_type = kind(program, "GGML_TYPE_F16")?;
    let bf16_type = kind(program, "GGML_TYPE_BF16")?;
    let i64_type = kind(program, "GGML_TYPE_I64")?;
    let i32_type = kind(program, "GGML_TYPE_I32")?;
    let i16_type = kind(program, "GGML_TYPE_I16")?;
    let i8_type = kind(program, "GGML_TYPE_I8")?;
    let primitive = [
        f32_type, f16_type, bf16_type, i64_type, i32_type, i16_type, i8_type,
    ]
    .contains(&tensor.kind);
    if primitive {
        if tensor.block_size != 1 {
            return Err(invalid("backend scalar type reports a non-unit block"));
        }
        tensor.offsets(|offset| {
            let value = if tensor.kind == f32_type {
                f32::from_le_bytes(element(&raw, offset)?)
            } else if tensor.kind == f16_type {
                fp16_to_float(u16::from_le_bytes(element(&raw, offset)?))
            } else if tensor.kind == bf16_type {
                f32::from_bits(u32::from(u16::from_le_bytes(element(&raw, offset)?)) << 16)
            } else if tensor.kind == i64_type {
                i64::from_le_bytes(element(&raw, offset)?) as f32
            } else if tensor.kind == i32_type {
                i32::from_le_bytes(element(&raw, offset)?) as f32
            } else if tensor.kind == i16_type {
                f32::from(i16::from_le_bytes(element(&raw, offset)?))
            } else {
                f32::from(i8::from_le_bytes(element(&raw, offset)?))
            };
            output.push(value);
            Ok(())
        })?;
    } else if elements != 0 {
        let observed = session.exchange(vec![json!({"Op":"type-info","Type":tensor.kind})])?;
        if observed[0]["Quantized"] != true || observed[0]["HasToFloat"] != true {
            return Err(invalid(
                "source tensor_to_float has no conversion for this production type",
            ));
        }
        let ordered = session.file(&format!("{artifact}-ordered.bin"))?;
        let decoded = session.file(&format!("{artifact}-decoded.bin"))?;
        let mut packed = BufWriter::new(
            File::options()
                .write(true)
                .create_new(true)
                .open(&ordered)?,
        );
        let width = usize::try_from(tensor.type_size)?;
        tensor.offsets(|offset| {
            let end = offset
                .checked_add(width)
                .ok_or_else(|| invalid("backend quantized block offset overflow"))?;
            packed
                .write_all(raw.get(offset..end).ok_or_else(|| {
                    invalid("backend quantized block exceeds raw tensor extent")
                })?)?;
            Ok(())
        })?;
        packed.flush()?;
        drop(packed);
        let observed = session.exchange(vec![json!({
            "Op":"dequantize-file","Type":tensor.kind,"Count":elements,
            "InputFile":ordered,"OutputFile":decoded
        })])?;
        let expected = u64::try_from(elements)?
            .checked_mul(4)
            .ok_or_else(|| invalid("backend decoded tensor extent overflow"))?;
        if observed[0]["Bytes"].as_u64() != Some(expected) {
            return Err(invalid("backend dequantizer response extent differs"));
        }
        let bytes = read_bytes(session, &decoded, expected)?;
        for bytes in bytes.as_chunks::<4>().0 {
            output.push(f32::from_le_bytes(*bytes));
        }
    }
    if output.len() != elements {
        return Err(invalid("source tensor traversal element count differs"));
    }
    Ok(output)
}
