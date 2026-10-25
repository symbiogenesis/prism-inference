// Copyright (c) 2023-2026 The ggml/llama.cpp authors
// SPDX-License-Identifier: MIT
//! Source input helpers. Only `random_device` entropy is replaced by a recorded seed ledger.

use super::super::{Result, invalid};
use super::graph::Plan;
use super::program::Program;
use super::random::{Mt19937, SeedLedger};
use super::tensor::Tensor;
use serde_json::{Value as Json, json};
use sha2::{Digest, Sha256};
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

fn write_floats(writer: &mut impl Write, values: &[f32]) -> Result<String> {
    let mut hash = Sha256::new();
    let mut bytes = vec![0_u8; 64 * 1024];
    for values in values.chunks(bytes.len() / 4) {
        for (value, destination) in values.iter().zip(bytes.as_chunks_mut::<4>().0) {
            destination.copy_from_slice(&value.to_le_bytes());
        }
        let encoded = &bytes[..values.len() * 4];
        writer.write_all(encoded)?;
        hash.update(encoded);
    }
    Ok(super::host::encode_hex(&hash.finalize()))
}

pub(super) struct Inputs {
    pub(super) ledger: SeedLedger,
    pub(super) receipts: Vec<Json>,
    pub(super) files: Vec<PathBuf>,
    pub(super) threads: usize,
    uniform_engine: Option<Mt19937>,
    serial: usize,
}

impl Inputs {
    pub(super) const fn new(seed: u64, threads: usize) -> Self {
        Self {
            ledger: SeedLedger::new(seed),
            receipts: Vec::new(),
            files: Vec::new(),
            threads,
            uniform_engine: None,
            serial: 0,
        }
    }

    pub(super) fn name(&mut self, plan: &Plan<'_>, purpose: &str) -> Result<PathBuf> {
        let serial = self.serial;
        self.serial += 1;
        let path = plan
            .session
            .file(&format!("input-{serial}-{purpose}.bin"))?;
        self.files.push(path.clone());
        Ok(path)
    }

    pub(super) fn record(&mut self, path: &Path, purpose: &str) -> Result<()> {
        let size = fs::metadata(path)?.len();
        if size > super::transport::MAX_FILE_BYTES {
            return Err(invalid("source input exceeds bounded file extent"));
        }
        let mut file = File::open(path)?;
        let mut hash = Sha256::new();
        let mut buffer = vec![0_u8; 64 * 1024];
        loop {
            let count = file.read(&mut buffer)?;
            if count == 0 {
                break;
            }
            hash.update(&buffer[..count]);
        }
        self.record_digest(
            path,
            purpose,
            size,
            &super::host::encode_hex(&hash.finalize()),
        )
    }

    fn record_digest(&mut self, path: &Path, purpose: &str, size: u64, digest: &str) -> Result<()> {
        self.receipts.push(json!({"File":path.file_name().and_then(|name|name.to_str()).ok_or_else(||invalid("source input name is not UTF-8"))?,"Bytes":size,"Sha256":digest,"Purpose":purpose}));
        Ok(())
    }

    pub(super) fn stage(
        &mut self,
        plan: &Plan<'_>,
        bytes: &[u8],
        purpose: &str,
    ) -> Result<PathBuf> {
        if bytes.len() as u64 > super::transport::MAX_FILE_BYTES {
            return Err(invalid("source staged input exceeds bounded extent"));
        }
        let path = self.name(plan, purpose)?;
        let mut file = File::options().create_new(true).write(true).open(&path)?;
        file.write_all(bytes)?;
        drop(file);
        self.record_digest(
            &path,
            purpose,
            bytes.len() as u64,
            &super::host::encode_hex(&Sha256::digest(bytes)),
        )?;
        Ok(path)
    }

    pub(super) fn stage_floats(
        &mut self,
        plan: &Plan<'_>,
        values: &[f32],
        purpose: &str,
    ) -> Result<PathBuf> {
        if values.len() as u64 * 4 > super::transport::MAX_FILE_BYTES {
            return Err(invalid("source float input exceeds bounded file extent"));
        }
        let path = self.name(plan, purpose)?;
        let mut file = File::options().create_new(true).write(true).open(&path)?;
        // Serialize and hash bounded blocks once. Per-scalar buffered writes
        // and a second full-file read dominate large source input tensors.
        let digest = write_floats(&mut file, values)?;
        drop(file);
        self.record_digest(&path, purpose, values.len() as u64 * 4, &digest)?;
        Ok(path)
    }

    pub(super) fn set_file(
        plan: &mut Plan<'_>,
        tensor: u32,
        path: &Path,
        offset: u64,
        bytes: u64,
    ) -> Result<()> {
        plan.push(json!({"Op":"tensor-set","Tensor":tensor,"File":path,"FileOffset":0,"TensorOffset":offset,"Bytes":bytes}));
        let _ = plan.flush()?;
        Ok(())
    }

    pub(super) fn set(
        &mut self,
        plan: &mut Plan<'_>,
        tensor: u32,
        bytes: &[u8],
        offset: u64,
        purpose: &str,
    ) -> Result<()> {
        let path = self.stage(plan, bytes, purpose)?;
        Self::set_file(plan, tensor, &path, offset, bytes.len() as u64)
    }

    fn info(plan: &mut Plan<'_>, kind: i32) -> Result<Json> {
        plan.push(json!({"Op":"type-info","Type":kind}));
        plan.flush()?
            .into_iter()
            .last()
            .ok_or_else(|| invalid("source type-info observation missing"))
    }

    pub(super) fn quantize(
        &mut self,
        plan: &mut Plan<'_>,
        tensor: &Tensor,
        values: &[f32],
        importance: Option<&[f32]>,
        purpose: &str,
    ) -> Result<PathBuf> {
        let input = self.stage_floats(plan, values, purpose)?;
        let output = self.name(plan, "quantized")?;
        let importance = importance
            .map(|values| self.stage_floats(plan, values, "importance"))
            .transpose()?;
        let block = usize::try_from(tensor.block_size)?;
        if block == 0 || !values.len().is_multiple_of(block) {
            return Err(invalid(
                "original quantization block alignment assertion failed",
            ));
        }
        plan.push(json!({"Op":"quantize-file","Type":tensor.kind,"InputFile":input,"OutputFile":output,"Start":0,"Rows":values.len()/block,"ElementsPerRow":block,"ImportanceFile":importance}));
        let observations = plan.flush()?;
        let observed = observations
            .last()
            .ok_or_else(|| invalid("source quantization observation missing"))?;
        let packed = values.len() as u64 / block as u64 * tensor.type_size;
        if observed["Written"].as_u64() != Some(packed) {
            return Err(invalid(
                "production quantized input byte extent differs from tensor",
            ));
        }
        self.record(&output, "production quantization of Rust input")?;
        Ok(output)
    }

    fn fill_uniform(&mut self, count: usize, minimum: f32, maximum: f32) -> Result<Vec<f32>> {
        let mut values = Vec::new();
        values.try_reserve_exact(count)?;
        values.resize(count, 0.0);
        if self.threads == 0 {
            return Err(invalid("source input thread count is zero"));
        }
        if self.threads == 1 {
            let engine = self
                .uniform_engine
                .get_or_insert_with(|| Mt19937::new(self.ledger.next()));
            for value in &mut values {
                *value = engine.uniform_real(minimum, maximum);
            }
        } else {
            for index in 0..self.threads {
                let mut engine = Mt19937::new(self.ledger.next());
                let start = index * count / self.threads;
                let end = (index + 1) * count / self.threads;
                for value in &mut values[start..end] {
                    *value = engine.uniform_real(minimum, maximum);
                }
            }
        }
        Ok(values)
    }

    #[allow(
        clippy::manual_midpoint,
        reason = "The original input selector uses a separate binary32 addition and multiplication, including its rounding/overflow behavior."
    )]
    pub(super) fn uniform(
        &mut self,
        plan: &mut Plan<'_>,
        program: &Program,
        id: u32,
        minimum: f32,
        maximum: f32,
    ) -> Result<()> {
        let tensor = plan.tensor(id)?;
        let count = usize::try_from(tensor.elements)?;
        let values = self.fill_uniform(count, minimum, maximum)?;
        let kind = |name: &str| {
            program.constants[name]
                .as_i64()
                .ok_or_else(|| invalid("source type constant differs"))
        };
        if i64::from(tensor.kind) == kind("GGML_TYPE_F32")?
            || i64::from(tensor.kind) == kind("GGML_TYPE_I32")?
        {
            let path = self.stage_floats(plan, &values, "uniform-float-bits")?;
            return Self::set_file(plan, id, &path, 0, count as u64 * 4);
        }
        let info = Self::info(plan, tensor.kind)?;
        let quantized = info["Quantized"]
            .as_bool()
            .ok_or_else(|| invalid("source quantized type observation differs"))?;
        if quantized
            || i64::from(tensor.kind) == kind("GGML_TYPE_F16")?
            || i64::from(tensor.kind) == kind("GGML_TYPE_BF16")?
        {
            let required = info["RequiresImportance"]
                .as_bool()
                .ok_or_else(|| invalid("source importance type observation differs"))?;
            let importance = vec![1.0_f32; usize::try_from(tensor.ne[0])?];
            let selected = required
                || values
                    .first()
                    .is_none_or(|&value| value <= 0.5_f32 * (minimum + maximum));
            let path = self.quantize(
                plan,
                &tensor,
                &values,
                selected.then_some(importance.as_slice()),
                "uniform-quantization",
            )?;
            return Self::set_file(
                plan,
                id,
                &path,
                0,
                count as u64 / tensor.block_size as u64 * tensor.type_size,
            );
        }
        let path = self.stage_floats(plan, &values, "uniform-integer-bits")?;
        if i64::from(tensor.kind) == kind("GGML_TYPE_I8")?
            || i64::from(tensor.kind) == kind("GGML_TYPE_I16")?
        {
            return Self::set_file(plan, id, &path, 0, count as u64 * tensor.type_size);
        }
        if i64::from(tensor.kind) == kind("GGML_TYPE_I64")? {
            Self::set_file(plan, id, &path, 0, count as u64 * 4)?;
            return Self::set_file(plan, id, &path, count as u64 * 4, count as u64 * 4);
        }
        Err(invalid(
            "original uniform initializer unsupported-type abort branch",
        ))
    }

    pub(super) fn mask(
        &mut self,
        plan: &mut Plan<'_>,
        program: &Program,
        id: u32,
        minimum: f32,
        maximum: f32,
    ) -> Result<()> {
        let tensor = plan.tensor(id)?;
        if program.constants["GGML_TYPE_F16"].as_i64() != Some(i64::from(tensor.kind)) {
            return Err(invalid("original KQ mask F16 assertion failed"));
        }
        let dimensions = tensor.ne.map(|value| value as i32);
        let [ne0, ne1, ne2, ne3] = dimensions;
        let count = usize::try_from(tensor.elements)?;
        let mut generator = Mt19937::new(self.ledger.next());
        let mut values = Vec::with_capacity(count);
        for _ in 0..count {
            values.push(generator.uniform_real(minimum, maximum));
        }
        let blocks = (0.2_f64
            * f64::from(ne0.wrapping_mul(ne1).wrapping_mul(ne2).wrapping_mul(ne3))
            / f64::from(128 * 64)) as i32;
        for _ in 0..blocks {
            let p3 = (self.ledger.next() % ne3 as u32) as i32;
            let p2 = (self.ledger.next() % ne2 as u32) as i32;
            let p1 = (self.ledger.next() % ne1 as u32) as i32;
            let p0 = (self.ledger.next() % ne0 as u32) as i32;
            let infinite = self.ledger.next() & 1 != 0;
            for i1 in 0..64.min(ne1 - p1) {
                let start = p3.wrapping_mul(ne2).wrapping_mul(ne1).wrapping_mul(ne0)
                    + p2.wrapping_mul(ne1).wrapping_mul(ne0)
                    + (p1 + i1).wrapping_mul(ne0)
                    + p0;
                for i0 in 0..128.min(ne0 - p0) {
                    let index = usize::try_from(start + i0)?;
                    *values
                        .get_mut(index)
                        .ok_or_else(|| invalid("source KQ mask block exceeds logical data"))? =
                        if infinite { f32::NEG_INFINITY } else { 0.0 };
                }
            }
        }
        let path = self.quantize(plan, &tensor, &values, None, "KQ-mask")?;
        Self::set_file(plan, id, &path, 0, count as u64 * 2)
    }

    pub(super) fn triangular(
        &mut self,
        plan: &mut Plan<'_>,
        program: &Program,
        id: u32,
        minimum: f32,
        maximum: f32,
    ) -> Result<()> {
        let tensor = plan.tensor(id)?;
        if program.constants["GGML_TYPE_F32"].as_i64() != Some(i64::from(tensor.kind))
            || tensor.ne[0] != tensor.ne[1]
        {
            return Err(invalid(
                "original triangular initializer type/shape assertion failed",
            ));
        }
        let mut values = vec![0.0_f32; usize::try_from(tensor.elements)?];
        let mut generator = Mt19937::new(self.ledger.next());
        for i3 in 0..tensor.ne[3] {
            for i2 in 0..tensor.ne[2] {
                for i1 in 0..tensor.ne[1] {
                    for i0 in 0..tensor.ne[0] {
                        let offset = (i0 as u64 * tensor.nb[0]
                            + i1 as u64 * tensor.nb[1]
                            + i2 as u64 * tensor.nb[2]
                            + i3 as u64 * tensor.nb[3])
                            / 4;
                        *values.get_mut(usize::try_from(offset)?).ok_or_else(|| {
                            invalid("source triangular index exceeds allocated values")
                        })? = if i0 <= i1 {
                            generator.uniform_real(minimum, maximum)
                        } else {
                            0.0
                        };
                    }
                }
            }
        }
        let path = self.stage_floats(plan, &values, "lower-triangular")?;
        Self::set_file(plan, id, &path, 0, tensor.bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn staged_float_blocks_preserve_all_bits_and_receipt_digest() -> Result<()> {
        for count in [0, 1, 16_383, 16_384, 16_385, 32_771] {
            let values = (0..count)
                .map(|index| {
                    f32::from_bits(
                        [
                            0,
                            0x8000_0000,
                            1,
                            0x7f80_0000,
                            0xff80_0000,
                            0x7fc0_1234,
                            0xff80_0042,
                            0x3f80_0000,
                        ][index % 8],
                    )
                })
                .collect::<Vec<_>>();
            let expected = values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>();
            let mut observed = Vec::new();
            let digest = write_floats(&mut observed, &values)?;
            assert_eq!(observed, expected);
            assert_eq!(
                digest,
                super::super::host::encode_hex(&Sha256::digest(&expected))
            );
        }
        Ok(())
    }
}
