//! Clean-room GGML session state, buffer storage, tensor/graph snapshots, and file actions.
//!
//! No `bonsai_prism_runtime`, `bonsai_gguf`, or native GGML libraries are linked or called.

use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Component, Path, PathBuf};
use std::time::Instant;

use serde_json::{Value, json};

use super::eval_ops;
use super::ops_builders;
use super::quants::{dequantize_to_f32, quantize_from_f32, type_info};
use super::types::{
    GGML_MAX_DIMS, GGML_MAX_SRC, TensorMeta, blck_size, compute_nbytes, decode_hex, i32_to_f32,
    read_i32_elem, read_unquant_elem_f32, type_size, write_i32_elem, write_unquant_elem_f32,
};
use crate::{Result, invalid};

#[derive(Clone, Debug)]
pub(super) enum ScalarValue {
    Bool(bool),
    I32(i32),
    I64(i64),
    U64(u64),
    String(String),
}

impl ScalarValue {
    pub(super) fn to_json(&self) -> Value {
        match self {
            Self::Bool(v) => json!({"Bool": v}),
            Self::I32(v) => json!({"I32": v}),
            Self::I64(v) => json!({"I64": v}),
            Self::U64(v) => json!({"U64": v}),
            Self::String(v) => json!({"String": v}),
        }
    }
}

pub(super) struct ContextRecord {
    pub(super) tensors: Vec<u32>,
    pub(super) buffer_id: Option<u32>,
}

pub(super) struct BackendRecord {
    pub(super) device: String,
}

pub(super) struct GraphRecord {
    pub(super) nodes: Vec<u32>,
    pub(super) leafs: Vec<u32>,
    pub(super) gradients: Vec<(u32, u32)>,
    pub(super) custom_size: usize,
}

pub(super) struct BackendState {
    pub(super) directory: PathBuf,
    pub(super) max_file_bytes: u64,
    pub(super) started_at: Instant,
    pub(super) contexts: BTreeMap<u32, ContextRecord>,
    pub(super) backends: BTreeMap<u32, BackendRecord>,
    pub(super) buffers: BTreeMap<u32, Vec<u8>>,
    pub(super) buffer_usage: BTreeMap<u32, i32>,
    pub(super) tensors: BTreeMap<u32, TensorMeta>,
    pub(super) aliases: BTreeMap<u32, u32>,
    pub(super) graphs: BTreeMap<u32, GraphRecord>,
    pub(super) scalars: BTreeMap<u64, ScalarValue>,
    pub(super) next_internal_id: u32,
}

impl BackendState {
    pub(super) fn new(directory: &Path, max_file_bytes: u64) -> Result<Self> {
        let canonical = fs::canonicalize(directory)?;
        if !canonical.is_absolute() || !canonical.is_dir() {
            return Err(invalid(
                "backend session directory must be existing absolute directory",
            ));
        }
        Ok(Self {
            directory: canonical,
            max_file_bytes,
            started_at: Instant::now(),
            contexts: BTreeMap::new(),
            backends: BTreeMap::new(),
            buffers: BTreeMap::new(),
            buffer_usage: BTreeMap::new(),
            tensors: BTreeMap::new(),
            aliases: BTreeMap::new(),
            graphs: BTreeMap::new(),
            scalars: BTreeMap::new(),
            next_internal_id: 1_000_000,
        })
    }

    pub(super) fn checked_path(&self, raw: &str, must_exist: bool) -> Result<PathBuf> {
        let candidate = Path::new(raw);
        if !candidate.is_absolute() {
            return Err(invalid("backend file path must be absolute"));
        }
        if candidate
            .components()
            .any(|part| matches!(part, Component::ParentDir))
        {
            return Err(invalid("backend file path cannot contain parent traversal"));
        }
        if must_exist {
            let resolved = fs::canonicalize(candidate)?;
            if !resolved.starts_with(&self.directory) || !resolved.is_file() {
                return Err(invalid("backend input file is outside session directory"));
            }
            return Ok(resolved);
        }
        let parent = candidate
            .parent()
            .ok_or_else(|| invalid("backend output file missing parent"))?;
        let resolved_parent = fs::canonicalize(parent)?;
        if !resolved_parent.starts_with(&self.directory) {
            return Err(invalid("backend output file escapes session directory"));
        }
        let file_name = candidate
            .file_name()
            .ok_or_else(|| invalid("backend output file missing name"))?;
        Ok(resolved_parent.join(file_name))
    }

    pub(super) fn canonical_id(&self, id: u32) -> Result<u32> {
        let cid = self.aliases.get(&id).copied().unwrap_or(id);
        if self.tensors.contains_key(&cid) {
            Ok(cid)
        } else {
            Err(invalid(format!("unknown tensor id {id}")))
        }
    }

    pub(super) fn tensor(&self, id: u32) -> Result<&TensorMeta> {
        let cid = self.canonical_id(id)?;
        self.tensors
            .get(&cid)
            .ok_or_else(|| invalid(format!("missing tensor {cid}")))
    }

    pub(super) fn tensor_mut(&mut self, id: u32) -> Result<&mut TensorMeta> {
        let cid = self.canonical_id(id)?;
        self.tensors
            .get_mut(&cid)
            .ok_or_else(|| invalid(format!("missing tensor {cid}")))
    }

    pub(super) fn register_alias(&mut self, existing: u32, alias: u32) -> Result<()> {
        let cid = self.canonical_id(existing)?;
        if alias != cid {
            self.aliases.insert(alias, cid);
        }
        if let Some(meta) = self.tensors.get_mut(&cid)
            && !meta.aliases.contains(&alias)
        {
            meta.aliases.push(alias);
            meta.aliases.sort_unstable();
        }
        Ok(())
    }

    pub(super) fn resolve_storage(&self, id: u32) -> Result<(u32, u64)> {
        let mut cur = self.canonical_id(id)?;
        let mut offset = 0_u64;
        for _ in 0..64 {
            let meta = self
                .tensors
                .get(&cur)
                .ok_or_else(|| invalid(format!("missing tensor {cur}")))?;
            if let Some(buf_id) = meta.buffer_id {
                let total = offset
                    .checked_add(meta.buffer_offset)
                    .ok_or_else(|| invalid("tensor buffer offset overflow"))?;
                return Ok((buf_id, total));
            }
            if let Some(parent) = meta.view_source {
                offset = offset
                    .checked_add(meta.view_offset)
                    .ok_or_else(|| invalid("tensor view offset overflow"))?;
                cur = parent;
            } else {
                return Err(invalid(format!("tensor {id} has no allocated buffer")));
            }
        }
        Err(invalid("tensor view chain exceeded depth limit"))
    }

    pub(super) fn read_raw(&self, id: u32, tensor_offset: u64, len: usize) -> Result<Vec<u8>> {
        if len == 0 {
            return Ok(Vec::new());
        }
        let (buf_id, base) = self.resolve_storage(id)?;
        let start = usize::try_from(
            base.checked_add(tensor_offset)
                .ok_or_else(|| invalid("tensor read offset overflow"))?,
        )?;
        let end = start
            .checked_add(len)
            .ok_or_else(|| invalid("tensor read end overflow"))?;
        let buf = self
            .buffers
            .get(&buf_id)
            .ok_or_else(|| invalid(format!("missing buffer {buf_id}")))?;
        let slice = buf
            .get(start..end)
            .ok_or_else(|| invalid("tensor read exceeds buffer extent"))?;
        Ok(slice.to_vec())
    }

    pub(super) fn write_raw(&mut self, id: u32, tensor_offset: u64, data: &[u8]) -> Result<()> {
        if data.is_empty() {
            return Ok(());
        }
        let (buf_id, base) = self.resolve_storage(id)?;
        let start = usize::try_from(
            base.checked_add(tensor_offset)
                .ok_or_else(|| invalid("tensor write offset overflow"))?,
        )?;
        let end = start
            .checked_add(data.len())
            .ok_or_else(|| invalid("tensor write end overflow"))?;
        let buf = self
            .buffers
            .get_mut(&buf_id)
            .ok_or_else(|| invalid(format!("missing buffer {buf_id}")))?;
        let dst = buf
            .get_mut(start..end)
            .ok_or_else(|| invalid("tensor write exceeds buffer extent"))?;
        dst.copy_from_slice(data);
        Ok(())
    }

    pub(super) fn read_elem_f32(&self, id: u32, idx: [i64; GGML_MAX_DIMS]) -> Result<f32> {
        let meta = self.tensor(id)?.clone();
        let (buf_id, base) = self.resolve_storage(id)?;
        let buf = self
            .buffers
            .get(&buf_id)
            .ok_or_else(|| invalid(format!("missing buffer {buf_id}")))?;
        let bs = blck_size(meta.kind);
        let ts = usize::try_from(type_size(meta.kind))?;
        let block_idx = idx[0].div_euclid(bs);
        let in_block = usize::try_from(idx[0].rem_euclid(bs))?;
        let rel = u64::try_from(block_idx)? * meta.nb[0]
            + u64::try_from(idx[1])? * meta.nb[1]
            + u64::try_from(idx[2])? * meta.nb[2]
            + u64::try_from(idx[3])? * meta.nb[3];
        let start = usize::try_from(base + rel)?;
        let raw = buf
            .get(start..start + ts)
            .ok_or_else(|| invalid("tensor element read exceeds buffer"))?;
        if bs == 1 {
            read_unquant_elem_f32(meta.kind, raw)
        } else {
            let block = dequantize_to_f32(meta.kind, raw, usize::try_from(bs)?)?;
            Ok(block[in_block])
        }
    }

    pub(super) fn write_elem_f32(
        &mut self,
        id: u32,
        idx: [i64; GGML_MAX_DIMS],
        val: f32,
    ) -> Result<()> {
        let meta = self.tensor(id)?.clone();
        let (buf_id, base) = self.resolve_storage(id)?;
        let bs = blck_size(meta.kind);
        let ts = usize::try_from(type_size(meta.kind))?;
        let rel = u64::try_from(idx[0].div_euclid(bs))? * meta.nb[0]
            + u64::try_from(idx[1])? * meta.nb[1]
            + u64::try_from(idx[2])? * meta.nb[2]
            + u64::try_from(idx[3])? * meta.nb[3];
        let start = usize::try_from(base + rel)?;
        let buf = self
            .buffers
            .get_mut(&buf_id)
            .ok_or_else(|| invalid(format!("missing buffer {buf_id}")))?;
        if bs == 1 {
            let dst = buf
                .get_mut(start..start + ts)
                .ok_or_else(|| invalid("tensor element write exceeds buffer"))?;
            write_unquant_elem_f32(meta.kind, dst, val)
        } else {
            let bs_usize = usize::try_from(bs)?;
            let in_block = usize::try_from(idx[0].rem_euclid(bs))?;
            let block_slice = buf
                .get(start..start + ts)
                .ok_or_else(|| invalid("quantized block read for update exceeds buffer"))?;
            let mut decoded = dequantize_to_f32(meta.kind, block_slice, bs_usize)?;
            decoded[in_block] = val;
            let encoded = quantize_from_f32(meta.kind, &decoded)?;
            let dst = buf
                .get_mut(start..start + ts)
                .ok_or_else(|| invalid("quantized block write exceeds buffer"))?;
            dst.copy_from_slice(&encoded);
            Ok(())
        }
    }

    pub(super) fn read_elem_i32(&self, id: u32, idx: [i64; GGML_MAX_DIMS]) -> Result<i32> {
        let meta = self.tensor(id)?.clone();
        let (buf_id, base) = self.resolve_storage(id)?;
        let buf = self
            .buffers
            .get(&buf_id)
            .ok_or_else(|| invalid(format!("missing buffer {buf_id}")))?;
        let ts = usize::try_from(type_size(meta.kind))?;
        let rel = u64::try_from(idx[0])? * meta.nb[0]
            + u64::try_from(idx[1])? * meta.nb[1]
            + u64::try_from(idx[2])? * meta.nb[2]
            + u64::try_from(idx[3])? * meta.nb[3];
        let start = usize::try_from(base + rel)?;
        let raw = buf
            .get(start..start + ts)
            .ok_or_else(|| invalid("tensor integer read exceeds buffer"))?;
        read_i32_elem(meta.kind, raw)
    }

    pub(super) fn write_elem_i32(
        &mut self,
        id: u32,
        idx: [i64; GGML_MAX_DIMS],
        val: i32,
    ) -> Result<()> {
        let meta = self.tensor(id)?.clone();
        let (buf_id, base) = self.resolve_storage(id)?;
        let ts = usize::try_from(type_size(meta.kind))?;
        let rel = u64::try_from(idx[0])? * meta.nb[0]
            + u64::try_from(idx[1])? * meta.nb[1]
            + u64::try_from(idx[2])? * meta.nb[2]
            + u64::try_from(idx[3])? * meta.nb[3];
        let start = usize::try_from(base + rel)?;
        let buf = self
            .buffers
            .get_mut(&buf_id)
            .ok_or_else(|| invalid(format!("missing buffer {buf_id}")))?;
        let dst = buf
            .get_mut(start..start + ts)
            .ok_or_else(|| invalid("tensor integer write exceeds buffer"))?;
        write_i32_elem(meta.kind, dst, val)
    }

    pub(super) fn read_all_f32(&self, id: u32) -> Result<(TensorMeta, Vec<f32>)> {
        let meta = self.tensor(id)?.clone();
        let count = usize::try_from(meta.nelements())?;
        let mut out = Vec::with_capacity(count);
        for d3 in 0..meta.ne[3] {
            for d2 in 0..meta.ne[2] {
                for d1 in 0..meta.ne[1] {
                    out.extend(self.read_row_f32(meta.id, d1, d2, d3)?);
                }
            }
        }
        Ok((meta, out))
    }

    pub(super) fn read_row_f32(&self, id: u32, d1: i64, d2: i64, d3: i64) -> Result<Vec<f32>> {
        let meta = self.tensor(id)?.clone();
        let (buf_id, base) = self.resolve_storage(id)?;
        let buf = self
            .buffers
            .get(&buf_id)
            .ok_or_else(|| invalid(format!("missing buffer {buf_id}")))?;
        let ne0 = usize::try_from(meta.ne[0])?;
        let bs = blck_size(meta.kind);
        let bs_usize = usize::try_from(bs)?;
        let ts = usize::try_from(type_size(meta.kind))?;
        let row_base = base
            + u64::try_from(d1)? * meta.nb[1]
            + u64::try_from(d2)? * meta.nb[2]
            + u64::try_from(d3)? * meta.nb[3];
        let mut out = Vec::with_capacity(ne0);
        if bs == 1 {
            if meta.nb[0] == type_size(meta.kind) && meta.kind == 0 {
                let start = usize::try_from(row_base)?;
                let byte_len = ne0 * 4;
                let slice = buf
                    .get(start..start + byte_len)
                    .ok_or_else(|| invalid("f32 row read exceeds buffer"))?;
                for chunk in slice.as_chunks::<4>().0 {
                    out.push(f32::from_le_bytes(*chunk));
                }
                return Ok(out);
            }
            for d0 in 0..meta.ne[0] {
                let start = usize::try_from(row_base + u64::try_from(d0)? * meta.nb[0])?;
                let raw = buf
                    .get(start..start + ts)
                    .ok_or_else(|| invalid("row element read exceeds buffer"))?;
                out.push(read_unquant_elem_f32(meta.kind, raw)?);
            }
            return Ok(out);
        }
        if meta.nb[0] == type_size(meta.kind) {
            let blocks = ne0 / bs_usize;
            let start = usize::try_from(row_base)?;
            let slice = buf
                .get(start..start + blocks * ts)
                .ok_or_else(|| invalid("quantized row read exceeds buffer"))?;
            return dequantize_to_f32(meta.kind, slice, ne0);
        }
        for blk in 0..(ne0 / bs_usize) {
            let start = usize::try_from(row_base + u64::try_from(blk)? * meta.nb[0])?;
            let slice = buf
                .get(start..start + ts)
                .ok_or_else(|| invalid("quantized block read exceeds buffer"))?;
            out.extend(dequantize_to_f32(meta.kind, slice, bs_usize)?);
        }
        Ok(out)
    }

    pub(super) fn write_row_f32(
        &mut self,
        id: u32,
        d1: i64,
        d2: i64,
        d3: i64,
        vals: &[f32],
    ) -> Result<()> {
        let meta = self.tensor(id)?.clone();
        let (buf_id, base) = self.resolve_storage(id)?;
        let ne0 = usize::try_from(meta.ne[0])?;
        if vals.len() != ne0 {
            return Err(invalid("write_row_f32 length mismatch"));
        }
        let bs = blck_size(meta.kind);
        let ts = usize::try_from(type_size(meta.kind))?;
        let row_base = base
            + u64::try_from(d1)? * meta.nb[1]
            + u64::try_from(d2)? * meta.nb[2]
            + u64::try_from(d3)? * meta.nb[3];
        if bs > 1 {
            let encoded = quantize_from_f32(meta.kind, vals)?;
            let bs_usize = usize::try_from(bs)?;
            let buf = self
                .buffers
                .get_mut(&buf_id)
                .ok_or_else(|| invalid(format!("missing buffer {buf_id}")))?;
            for blk in 0..(ne0 / bs_usize) {
                let start = usize::try_from(row_base + u64::try_from(blk)? * meta.nb[0])?;
                let dst = buf
                    .get_mut(start..start + ts)
                    .ok_or_else(|| invalid("quantized row write exceeds buffer"))?;
                dst.copy_from_slice(&encoded[blk * ts..(blk + 1) * ts]);
            }
            return Ok(());
        }
        let buf = self
            .buffers
            .get_mut(&buf_id)
            .ok_or_else(|| invalid(format!("missing buffer {buf_id}")))?;
        for (idx, &val) in vals.iter().enumerate() {
            let start = usize::try_from(row_base + u64::try_from(idx)? * meta.nb[0])?;
            let dst = buf
                .get_mut(start..start + ts)
                .ok_or_else(|| invalid("row element write exceeds buffer"))?;
            write_unquant_elem_f32(meta.kind, dst, val)?;
        }
        Ok(())
    }

    pub(super) fn run_action(&mut self, action: &Value) -> Result<Value> {
        let op = action
            .get("Op")
            .and_then(Value::as_str)
            .ok_or_else(|| invalid("backend action missing Op string"))?;
        match op {
            "context-init" => self.op_context_init(action),
            "backend-init" => self.op_backend_init(action),
            "backend-features" => self.op_backend_features(action),
            "supports" => self.op_supports(action),
            "allocate" => self.op_allocate(action),
            "call" => self.op_call(action),
            "scalar" => self.op_scalar(action),
            "tensor-set" => self.op_tensor_set(action),
            "tensor-get" => self.op_tensor_get(action),
            "tensor-write" => self.op_tensor_write(action),
            "tensor-snapshot" => self.op_tensor_snapshot(action),
            "context-snapshot" => self.op_context_snapshot(action),
            "graph-snapshot" => self.op_graph_snapshot(action),
            "clock" => self.op_clock(),
            "type-info" => Self::op_type_info(action),
            "quantize-file" => self.op_quantize_file(action),
            "dequantize-file" => self.op_dequantize_file(action),
            "compare" => self.op_compare(action),
            _ => Err(invalid(format!("unknown backend action Op {op}"))),
        }
    }

    fn op_context_init(&mut self, action: &Value) -> Result<Value> {
        let id = u32::try_from(req_u64(action, "Id")?)?;
        let _memory_bytes = req_u64(action, "MemoryBytes")?;
        let _no_alloc = action
            .get("NoAlloc")
            .and_then(Value::as_bool)
            .ok_or_else(|| invalid("context-init missing NoAlloc"))?;
        self.contexts.insert(
            id,
            ContextRecord {
                tensors: Vec::new(),
                buffer_id: None,
            },
        );
        Ok(json!({}))
    }

    fn op_backend_init(&mut self, action: &Value) -> Result<Value> {
        let id = u32::try_from(req_u64(action, "Id")?)?;
        let device = action
            .get("Device")
            .and_then(Value::as_str)
            .ok_or_else(|| invalid("backend-init missing Device"))?
            .to_owned();
        let _threads = match action.get("Threads") {
            Some(Value::Null) => 0_u32,
            Some(v) => {
                let n = v
                    .as_u64()
                    .ok_or_else(|| invalid("backend-init Threads must be u64 or null"))?;
                if !(1..=256).contains(&n) {
                    return Err(invalid("backend-init Threads out of range 1..=256"));
                }
                u32::try_from(n)?
            }
            None => return Err(invalid("backend-init missing Threads")),
        };
        let _use_ref = action
            .get("UseRef")
            .and_then(Value::as_bool)
            .ok_or_else(|| invalid("backend-init missing UseRef"))?;
        self.backends.insert(id, BackendRecord { device });
        Ok(json!({"UseRefAvailable": true}))
    }

    fn op_backend_features(&self, action: &Value) -> Result<Value> {
        let backend_id = u32::try_from(req_u64(action, "Backend")?)?;
        let backend = self
            .backends
            .get(&backend_id)
            .ok_or_else(|| invalid("backend-features unknown backend"))?;
        Ok(json!({"RegistryName": backend.device, "Features": []}))
    }

    fn op_supports(&self, action: &Value) -> Result<Value> {
        let backend_id = u32::try_from(req_u64(action, "Backend")?)?;
        let tensor_id = u32::try_from(req_u64(action, "Tensor")?)?;
        if !self.backends.contains_key(&backend_id) {
            return Err(invalid("supports unknown backend"));
        }
        let tensor = self.tensor(tensor_id)?;
        let (_, _, has_float) = type_info(tensor.kind);
        Ok(json!({"Supported": has_float}))
    }

    fn op_allocate(&mut self, action: &Value) -> Result<Value> {
        let context_id = u32::try_from(req_u64(action, "Context")?)?;
        let backend_id = u32::try_from(req_u64(action, "Backend")?)?;
        let buf_id = u32::try_from(
            action
                .get("Result")
                .and_then(|v| v.get("Buffer"))
                .and_then(Value::as_u64)
                .ok_or_else(|| invalid("allocate missing Result.Buffer"))?,
        )?;
        if !self.backends.contains_key(&backend_id) {
            return Err(invalid("allocate unknown backend"));
        }
        let tensor_ids = self
            .contexts
            .get(&context_id)
            .ok_or_else(|| invalid("allocate unknown context"))?
            .tensors
            .clone();
        let mut cursor = 0_u64;
        for tid in tensor_ids {
            let meta = self
                .tensors
                .get_mut(&tid)
                .ok_or_else(|| invalid("allocate missing tensor in context"))?;
            if meta.view_source.is_some() || meta.buffer_id.is_some() {
                continue;
            }
            let size = meta.bytes.max(compute_nbytes(meta.kind, meta.ne, meta.nb)?);
            let aligned = cursor.div_ceil(16) * 16;
            meta.buffer_id = Some(buf_id);
            meta.buffer_offset = aligned;
            cursor = aligned
                .checked_add(size.max(16))
                .ok_or_else(|| invalid("allocate buffer cursor overflow"))?;
        }
        let total = usize::try_from(cursor)?;
        self.buffers.insert(buf_id, vec![0_u8; total]);
        self.buffer_usage.insert(buf_id, 0);
        if let Some(ctx) = self.contexts.get_mut(&context_id) {
            ctx.buffer_id = Some(buf_id);
        }
        Ok(json!({}))
    }

    fn op_call(&mut self, action: &Value) -> Result<Value> {
        let func = action
            .get("Function")
            .and_then(Value::as_str)
            .ok_or_else(|| invalid("call missing Function"))?;
        let args = action
            .get("Arguments")
            .and_then(Value::as_array)
            .ok_or_else(|| invalid("call missing Arguments array"))?;
        let result = action.get("Result").unwrap_or(&Value::Null);
        ops_builders::dispatch(self, func, args, result)?;
        Ok(json!({}))
    }

    fn op_scalar(&mut self, action: &Value) -> Result<Value> {
        let id = req_u64(action, "Scalar")?;
        let release = action
            .get("Release")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let val = if release {
            self.scalars
                .remove(&id)
                .ok_or_else(|| invalid(format!("unknown scalar id {id}")))?
        } else {
            self.scalars
                .get(&id)
                .cloned()
                .ok_or_else(|| invalid(format!("unknown scalar id {id}")))?
        };
        Ok(json!({"Value": val.to_json()}))
    }

    fn op_tensor_set(&mut self, action: &Value) -> Result<Value> {
        let tid = u32::try_from(req_u64(action, "Tensor")?)?;
        let path_str = req_str(action, "File")?;
        let file_off = req_u64(action, "FileOffset")?;
        let tensor_off = req_u64(action, "TensorOffset")?;
        let bytes = req_u64(action, "Bytes")?;
        if bytes > self.max_file_bytes {
            return Err(invalid("tensor-set bytes exceed max file bytes"));
        }
        let path = self.checked_path(path_str, true)?;
        let len = usize::try_from(bytes)?;
        let meta = fs::metadata(&path)?;
        if file_off.saturating_add(bytes) > meta.len() {
            return Err(invalid("tensor-set file read exceeds file length"));
        }
        let mut buf = vec![0_u8; len];
        let mut file = File::open(path)?;
        file.seek(SeekFrom::Start(file_off))?;
        file.read_exact(&mut buf)?;
        self.write_raw(tid, tensor_off, &buf)?;
        Ok(json!({}))
    }

    fn op_tensor_get(&self, action: &Value) -> Result<Value> {
        let tid = u32::try_from(req_u64(action, "Tensor")?)?;
        let path_str = req_str(action, "File")?;
        let file_off = req_u64(action, "FileOffset")?;
        let tensor_off = req_u64(action, "TensorOffset")?;
        let bytes = req_u64(action, "Bytes")?;
        if bytes > self.max_file_bytes {
            return Err(invalid("tensor-get bytes exceed max file bytes"));
        }
        let truncate = action
            .get("Truncate")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let path = self.checked_path(path_str, false)?;
        let raw = self.read_raw(tid, tensor_off, usize::try_from(bytes)?)?;
        let mut file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(truncate)
            .open(path)?;
        file.seek(SeekFrom::Start(file_off))?;
        file.write_all(&raw)?;
        file.flush()?;
        Ok(json!({}))
    }

    fn op_tensor_write(&mut self, action: &Value) -> Result<Value> {
        let tid = u32::try_from(req_u64(action, "Tensor")?)?;
        let fields = action
            .get("Fields")
            .and_then(Value::as_object)
            .ok_or_else(|| invalid("tensor-write missing Fields object"))?;
        let sources_mapped = if let Some(srcs) = fields.get("Sources").and_then(Value::as_array) {
            if srcs.len() > GGML_MAX_SRC {
                return Err(invalid("tensor-write too many sources"));
            }
            let mut arr = [None; GGML_MAX_SRC];
            for (idx, val) in srcs.iter().enumerate() {
                arr[idx] = match val {
                    Value::Null => None,
                    other => Some(self.canonical_id(u32::try_from(
                        other.as_u64().ok_or_else(|| invalid("invalid source id"))?,
                    )?)?),
                };
            }
            Some(arr)
        } else {
            None
        };
        let op_val = fields
            .get("Op")
            .and_then(Value::as_i64)
            .map(i32::try_from)
            .transpose()?;
        let nb_val = if let Some(nb_arr) = fields.get("NB").and_then(Value::as_array) {
            if nb_arr.len() != GGML_MAX_DIMS {
                return Err(invalid("tensor-write NB must have 4 elements"));
            }
            let mut parsed = [0_u64; GGML_MAX_DIMS];
            for (idx, v) in nb_arr.iter().enumerate() {
                parsed[idx] = v
                    .as_u64()
                    .ok_or_else(|| invalid("tensor-write NB element must be u64"))?;
            }
            Some(parsed)
        } else {
            None
        };
        let op_params_hex = fields
            .get("OpParamsHex")
            .and_then(Value::as_str)
            .map(decode_hex)
            .transpose()?;
        let meta = self.tensor_mut(tid)?;
        if let Some(op_code) = op_val {
            meta.op = op_code;
        }
        if let Some(nb) = nb_val {
            meta.nb = nb;
            meta.refresh_bytes()?;
        }
        if let Some(srcs) = sources_mapped {
            meta.sources = srcs;
        }
        if let Some(bytes) = op_params_hex {
            if bytes.len() > meta.op_params.len() {
                return Err(invalid("tensor-write OpParamsHex exceeds 64 bytes"));
            }
            meta.op_params.fill(0);
            meta.op_params[..bytes.len()].copy_from_slice(&bytes);
        }
        Ok(json!({}))
    }

    fn op_tensor_snapshot(&self, action: &Value) -> Result<Value> {
        let tid = u32::try_from(req_u64(action, "Tensor")?)?;
        let tensor = self.tensor(tid)?;
        Ok(json!({"Tensor": tensor.to_snapshot_json()}))
    }

    fn op_context_snapshot(&self, action: &Value) -> Result<Value> {
        let cid = u32::try_from(req_u64(action, "Context")?)?;
        let ctx = self
            .contexts
            .get(&cid)
            .ok_or_else(|| invalid("context-snapshot unknown context"))?;
        let mut snapshots = Vec::with_capacity(ctx.tensors.len());
        for &tid in &ctx.tensors {
            let t = self
                .tensors
                .get(&tid)
                .ok_or_else(|| invalid("context-snapshot missing tensor"))?;
            snapshots.push(t.to_snapshot_json());
        }
        Ok(json!({"Tensors": snapshots}))
    }

    fn op_graph_snapshot(&self, action: &Value) -> Result<Value> {
        let gid = u32::try_from(req_u64(action, "Graph")?)?;
        let graph = self
            .graphs
            .get(&gid)
            .ok_or_else(|| invalid("graph-snapshot unknown graph"))?;
        let grads = graph
            .gradients
            .iter()
            .map(|(tensor, grad)| json!({"Tensor": tensor, "Gradient": grad}))
            .collect::<Vec<_>>();
        Ok(json!({"Nodes": graph.nodes, "Gradients": grads}))
    }

    fn op_clock(&self) -> Result<Value> {
        let micros = i64::try_from(self.started_at.elapsed().as_micros())?;
        Ok(json!({"Microseconds": micros}))
    }

    fn op_type_info(action: &Value) -> Result<Value> {
        let kind = i32::try_from(req_i64(action, "Type")?)?;
        let (quantized, requires_importance, has_to_float) = type_info(kind);
        Ok(json!({
            "Quantized": quantized,
            "RequiresImportance": requires_importance,
            "HasToFloat": has_to_float,
        }))
    }

    fn op_quantize_file(&self, action: &Value) -> Result<Value> {
        let kind = i32::try_from(req_i64(action, "Type")?)?;
        let input = self.checked_path(req_str(action, "InputFile")?, true)?;
        let output = self.checked_path(req_str(action, "OutputFile")?, false)?;
        let raw = fs::read(&input)?;
        if u64::try_from(raw.len())? > self.max_file_bytes {
            return Err(invalid("quantize-file input exceeds bound"));
        }
        let mut floats = Vec::with_capacity(raw.len() / 4);
        for chunk in raw.as_chunks::<4>().0 {
            floats.push(f32::from_le_bytes(*chunk));
        }
        let encoded = quantize_from_f32(kind, &floats)?;
        fs::write(&output, &encoded)?;
        Ok(json!({"Written": u64::try_from(encoded.len())?}))
    }

    fn op_dequantize_file(&self, action: &Value) -> Result<Value> {
        let kind = i32::try_from(req_i64(action, "Type")?)?;
        let count = usize::try_from(req_u64(action, "Count")?)?;
        let input = self.checked_path(req_str(action, "InputFile")?, true)?;
        let output = self.checked_path(req_str(action, "OutputFile")?, false)?;
        let raw = fs::read(&input)?;
        let floats = dequantize_to_f32(kind, &raw, count)?;
        let mut out_bytes = Vec::with_capacity(floats.len() * 4);
        for v in floats {
            out_bytes.extend_from_slice(&v.to_le_bytes());
        }
        fs::write(&output, &out_bytes)?;
        Ok(json!({"Bytes": u64::try_from(out_bytes.len())?}))
    }

    fn op_compare(&mut self, action: &Value) -> Result<Value> {
        let backend1 = u32::try_from(req_u64(action, "Backend1")?)?;
        let backend2 = u32::try_from(req_u64(action, "Backend2")?)?;
        let graph_id = u32::try_from(req_u64(action, "Graph")?)?;
        let prefix = req_str(action, "Prefix")?;
        let max_output = action
            .get("MaxOutputBytes")
            .and_then(Value::as_u64)
            .unwrap_or(self.max_file_bytes);
        let selected: Option<Vec<u32>> = match action.get("Nodes") {
            Some(Value::Array(arr)) => {
                let mut list = Vec::with_capacity(arr.len());
                for item in arr {
                    let raw_id = u32::try_from(
                        item.as_u64()
                            .ok_or_else(|| invalid("compare Nodes element must be integer"))?,
                    )?;
                    list.push(self.canonical_id(raw_id)?);
                }
                Some(list)
            }
            _ => None,
        };

        let graph_nodes = self
            .graphs
            .get(&graph_id)
            .ok_or_else(|| invalid("compare unknown graph"))?
            .nodes
            .clone();
        let captured: Vec<(usize, u32)> = graph_nodes
            .iter()
            .enumerate()
            .filter_map(|(idx, &tid)| {
                let _ = self.tensors.get(&tid)?;
                let include = selected.as_ref().is_none_or(|sel| sel.contains(&tid));
                if include { Some((idx, tid)) } else { None }
            })
            .collect();

        let saved_buffers = self.buffers.clone();
        eval_ops::eval_graph(self, backend1, graph_id)?;
        let mut left_captures = Vec::with_capacity(captured.len());
        for &(_, tid) in &captured {
            let meta = self.tensor(tid)?.clone();
            let raw = self.read_raw(tid, 0, usize::try_from(meta.bytes)?)?;
            left_captures.push((meta, raw));
        }

        self.buffers = saved_buffers;
        eval_ops::eval_graph(self, backend2, graph_id)?;

        let mut pairs = Vec::with_capacity(captured.len());
        let mut total_written = 0_u64;
        for (ordinal, (&(node_idx, tid), (left_meta, left_raw))) in
            captured.iter().zip(left_captures).enumerate()
        {
            let right_meta = self.tensor(tid)?.clone();
            let right_raw = self.read_raw(tid, 0, usize::try_from(right_meta.bytes)?)?;
            let step_bytes = u64::try_from(left_raw.len() + right_raw.len())?;
            total_written = total_written
                .checked_add(step_bytes)
                .ok_or_else(|| invalid("compare output bytes overflow"))?;
            if total_written > max_output {
                return Err(invalid("compare output exceeds MaxOutputBytes"));
            }
            let left_file = self.checked_path(&format!("{prefix}-{ordinal}-left.bin"), false)?;
            let right_file = self.checked_path(&format!("{prefix}-{ordinal}-right.bin"), false)?;
            fs::write(&left_file, &left_raw)?;
            fs::write(&right_file, &right_raw)?;
            pairs.push(json!({
                "Ordinal": ordinal,
                "Index": node_idx,
                "Left": left_meta.to_snapshot_json(),
                "Right": right_meta.to_snapshot_json(),
                "LeftFile": left_file,
                "RightFile": right_file,
            }));
        }
        Ok(json!({"Compared": true, "Pairs": pairs}))
    }
}

pub(super) fn alloc_tensor(
    state: &mut BackendState,
    context_id: u32,
    new_id: u32,
    kind: i32,
    ne: [i64; GGML_MAX_DIMS],
) -> Result<TensorMeta> {
    let ctx = state
        .contexts
        .get_mut(&context_id)
        .ok_or_else(|| invalid(format!("unknown context {context_id}")))?;
    let meta = TensorMeta::new_base(new_id, kind, ne)?;
    ctx.tensors.push(new_id);
    state.tensors.insert(new_id, meta.clone());
    Ok(meta)
}

pub(super) fn allocate_tensor_zero_fill(state: &mut BackendState, id: u32, val: f32) -> Result<()> {
    let meta = state.tensor(id)?.clone();
    if meta.buffer_id.is_none() && meta.view_source.is_none() {
        return Ok(());
    }
    for d3 in 0..meta.ne[3] {
        for d2 in 0..meta.ne[2] {
            for d1 in 0..meta.ne[1] {
                let ne0 = usize::try_from(meta.ne[0])?;
                let row = vec![val; ne0];
                state.write_row_f32(id, d1, d2, d3, &row)?;
            }
        }
    }
    Ok(())
}

pub(super) fn set_tensor_zero(state: &mut BackendState, id: u32) -> Result<()> {
    allocate_tensor_zero_fill(state, id, 0.0)
}

fn req_u64(val: &Value, key: &str) -> Result<u64> {
    val.get(key)
        .and_then(Value::as_u64)
        .ok_or_else(|| invalid(format!("action missing u64 field {key}")))
}

fn req_i64(val: &Value, key: &str) -> Result<i64> {
    val.get(key)
        .and_then(Value::as_i64)
        .ok_or_else(|| invalid(format!("action missing i64 field {key}")))
}

fn req_str<'a>(val: &'a Value, key: &str) -> Result<&'a str> {
    val.get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| invalid(format!("action missing string field {key}")))
}

pub(super) fn arg_handle(arg: &Value, key: &str) -> Result<u32> {
    arg.get(key)
        .and_then(Value::as_u64)
        .ok_or_else(|| invalid(format!("argument missing handle {key}")))
        .and_then(|v| Ok(u32::try_from(v)?))
}

pub(super) fn arg_opt_tensor(arg: &Value) -> Result<Option<u32>> {
    if arg.is_null() {
        return Ok(None);
    }
    arg_handle(arg, "Tensor").map(Some)
}

pub(super) fn arg_i32(arg: &Value) -> Result<i32> {
    if let Some(v) = arg.get("I32").and_then(Value::as_i64) {
        return Ok(i32::try_from(v)?);
    }
    if let Some(v) = arg.get("I64").and_then(Value::as_i64) {
        return Ok(i32::try_from(v)?);
    }
    if let Some(v) = arg.get("U64").and_then(Value::as_u64) {
        return Ok(i32::try_from(v)?);
    }
    if let Some(v) = arg.get("Bool").and_then(Value::as_bool) {
        return Ok(i32::from(v));
    }
    if let Some(v) = arg.as_i64() {
        return Ok(i32::try_from(v)?);
    }
    Err(invalid("argument is not an i32"))
}

pub(super) fn arg_i64(arg: &Value) -> Result<i64> {
    if let Some(v) = arg.get("I64").and_then(Value::as_i64) {
        return Ok(v);
    }
    if let Some(v) = arg.get("I32").and_then(Value::as_i64) {
        return Ok(v);
    }
    if let Some(v) = arg.get("U64").and_then(Value::as_u64) {
        return Ok(i64::try_from(v)?);
    }
    if let Some(v) = arg.as_i64() {
        return Ok(v);
    }
    Err(invalid("argument is not an i64"))
}

pub(super) fn arg_u64(arg: &Value) -> Result<u64> {
    if let Some(v) = arg.get("U64").and_then(Value::as_u64) {
        return Ok(v);
    }
    if let Some(v) = arg.get("I64").and_then(Value::as_i64) {
        return Ok(u64::try_from(v)?);
    }
    if let Some(v) = arg.get("I32").and_then(Value::as_i64) {
        return Ok(u64::try_from(v)?);
    }
    if let Some(v) = arg.as_u64() {
        return Ok(v);
    }
    Err(invalid("argument is not a u64"))
}

pub(super) fn arg_bool(arg: &Value) -> Result<bool> {
    if let Some(v) = arg.get("Bool").and_then(Value::as_bool) {
        return Ok(v);
    }
    if let Some(v) = arg.as_bool() {
        return Ok(v);
    }
    Err(invalid("argument is not a bool"))
}

pub(super) fn arg_f32(arg: &Value) -> Result<f32> {
    if let Some(bits) = arg.get("F32Bits").and_then(Value::as_u64) {
        return Ok(f32::from_bits(u32::try_from(bits)?));
    }
    if let Some(bits) = arg.get("F64Bits").and_then(Value::as_u64) {
        return Ok(super::types::f64_to_f32(f64::from_bits(bits)));
    }
    if let Some(v) = arg.get("I32").and_then(Value::as_i64) {
        return Ok(i32_to_f32(i32::try_from(v)?));
    }
    if let Some(v) = arg.get("I64").and_then(Value::as_i64) {
        return Ok(super::types::i64_to_f32(v));
    }
    if let Some(v) = arg.as_f64() {
        return Ok(super::types::f64_to_f32(v));
    }
    Err(invalid("argument is not an f32"))
}

pub(super) fn arg_str(arg: &Value) -> Result<&str> {
    if let Some(s) = arg.get("String").and_then(Value::as_str) {
        return Ok(s);
    }
    if let Some(s) = arg.as_str() {
        return Ok(s);
    }
    Err(invalid("argument is not a string"))
}

pub(super) fn arg_i32_array(arg: &Value) -> Result<Vec<i32>> {
    let arr = arg
        .get("I32Array")
        .or_else(|| arg.get("I64Array"))
        .and_then(Value::as_array)
        .ok_or_else(|| invalid("argument is not an integer array"))?;
    let mut out = Vec::with_capacity(arr.len());
    for item in arr {
        let v = item
            .as_i64()
            .ok_or_else(|| invalid("array element must be integer"))?;
        out.push(i32::try_from(v)?);
    }
    Ok(out)
}

pub(super) fn arg_i64_array(arg: &Value) -> Result<Vec<i64>> {
    let arr = arg
        .get("I64Array")
        .or_else(|| arg.get("I32Array"))
        .and_then(Value::as_array)
        .ok_or_else(|| invalid("argument is not an i64 array"))?;
    let mut out = Vec::with_capacity(arr.len());
    for item in arr {
        let v = item
            .as_i64()
            .ok_or_else(|| invalid("array element must be integer"))?;
        out.push(v);
    }
    Ok(out)
}

pub(super) fn arg_tensor_array(arg: &Value) -> Result<Vec<Option<u32>>> {
    let arr = arg
        .get("TensorPtrArray")
        .or_else(|| arg.get("TensorArray"))
        .and_then(Value::as_array)
        .ok_or_else(|| invalid("argument is not a tensor pointer array"))?;
    let mut out = Vec::with_capacity(arr.len());
    for item in arr {
        if item.is_null() {
            out.push(None);
        } else if let Some(id) = item.as_u64() {
            out.push(Some(u32::try_from(id)?));
        } else if let Some(id) = item.get("Tensor").and_then(Value::as_u64) {
            out.push(Some(u32::try_from(id)?));
        } else {
            return Err(invalid("invalid entry in tensor pointer array"));
        }
    }
    Ok(out)
}
