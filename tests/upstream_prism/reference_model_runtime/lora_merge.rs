//! Clean-room in-process `GGUF` `LoRA` adapter merge (`export_lora_in_process`).
//!
//! No `bonsai_prism_runtime`, `bonsai_gguf`, or external `llama-export-lora` executable is used.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::BufWriter;
use std::path::Path;

use serde_json::{Value, json};

use super::gguf_model::{GgufValue, ModelData, TensorEntry, f64_to_f32};
use crate::{Result, invalid};

const TYPE_F32: i32 = 0;

pub(super) fn export_lora_in_process(
    base: &Path,
    adapter: &Path,
    output: &Path,
    _directory: &Path,
) -> Result<Value> {
    let base_arc = ModelData::load_from_file(base, true, None, None)?
        .ok_or_else(|| invalid("base GGUF load cancelled"))?;
    let adapter_arc = ModelData::load_from_file(adapter, true, None, None)?
        .ok_or_else(|| invalid("adapter GGUF load cancelled"))?;
    let alpha = adapter_arc
        .kvs
        .get("general.lora.alpha")
        .and_then(GgufValue::as_f32_coerce)
        .unwrap_or(1.0);
    let mut merged = (*base_arc).clone();
    let pairs = collect_lora_pairs(&adapter_arc)?;
    for (target_name, (a_name, b_name)) in pairs {
        if let Some(entry) = merged.tensors.get_mut(&target_name) {
            merge_one_tensor(entry, &adapter_arc, &a_name, &b_name, alpha)?;
        }
    }
    let out_file = File::create(output)?;
    let mut writer = BufWriter::new(out_file);
    merged.save_to_writer(&mut writer)?;
    drop(writer);
    let bytes = fs::read(output)?;
    let length = u64::try_from(bytes.len())?;
    if !(24..=256 * 1024 * 1024).contains(&length) {
        return Err(invalid("merged production GGUF extent differs"));
    }
    let sha = hex::encode(bonsai_native_hash::hash_bytes(&bytes)?);
    Ok(json!({
        "Tool": "in-process-reference-lora-merge",
        "Status": 0,
        "Bytes": length,
        "Sha256": sha,
        "StdoutBytes": 0,
        "StderrBytes": 0,
    }))
}

fn collect_lora_pairs(adapter: &ModelData) -> Result<BTreeMap<String, (String, String)>> {
    let mut bases = BTreeSet::new();
    for name in &adapter.tensor_order {
        if let Some(stem) = name.strip_suffix(".lora_a") {
            bases.insert(stem.to_owned());
        } else if let Some(stem) = name.strip_suffix(".lora_b") {
            bases.insert(stem.to_owned());
        }
    }
    let mut pairs = BTreeMap::new();
    for stem in bases {
        let a_name = format!("{stem}.lora_a");
        let b_name = format!("{stem}.lora_b");
        if !adapter.tensors.contains_key(&a_name) || !adapter.tensors.contains_key(&b_name) {
            return Err(invalid(format!("incomplete LoRA factor pair for {stem}")));
        }
        pairs.insert(stem, (a_name, b_name));
    }
    Ok(pairs)
}

fn merge_one_tensor(
    entry: &mut TensorEntry,
    adapter: &ModelData,
    a_name: &str,
    b_name: &str,
    alpha: f32,
) -> Result<()> {
    let a = &adapter.tensors[a_name];
    let b = &adapter.tensors[b_name];
    let in_dim = usize::try_from(a.ne[0])?;
    let rank = usize::try_from(a.ne[1])?;
    let out_dim = usize::try_from(b.ne[1])?;
    let b_dim0 = usize::try_from(b.ne[0])?;
    let w_in = usize::try_from(entry.ne[0])?;
    let w_out = usize::try_from(entry.ne[1])?;
    if rank == 0 || b_dim0 != rank || w_in != in_dim || w_out != out_dim {
        return Err(invalid(
            "LoRA factor tensor dimensions mismatch base weight",
        ));
    }
    let scale = f64::from(alpha) / f64::from(u32::try_from(rank)?);
    let a_vals = a.to_f32()?;
    let b_vals = b.to_f32()?;
    let mut w_vals = entry.to_f32()?;
    for row in 0..out_dim {
        for col in 0..in_dim {
            let mut dot = 0.0_f64;
            for r in 0..rank {
                dot = f64::mul_add(
                    f64::from(b_vals[row * rank + r]),
                    f64::from(a_vals[r * in_dim + col]),
                    dot,
                );
            }
            w_vals[row * in_dim + col] =
                f64_to_f32(dot.mul_add(scale, f64::from(w_vals[row * in_dim + col])));
        }
    }
    let mut raw = Vec::with_capacity(w_vals.len() * 4);
    for val in w_vals {
        raw.extend_from_slice(&val.to_bits().to_le_bytes());
    }
    entry.kind = TYPE_F32;
    entry.raw = raw;
    Ok(())
}
