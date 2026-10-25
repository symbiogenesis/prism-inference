// SPDX-License-Identifier: MIT
//! Production observations for the Rust source interpreter; all decisions remain in Rust.

use super::super::{Result, invalid};
use super::graph::Plan;
use super::inputs::Inputs;
use super::interpreter::Host;
use super::metrics;
use super::number::{Kind, Number};
use super::program::Program;
use super::random::CrtRandom;
use super::value::{Buffer, Element, Handle, Value};
use serde_json::{Value as Json, json};
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::fs;
use std::rc::Rc;

pub(super) struct Production<'a, 'b> {
    pub(super) graph: Plan<'a>,
    pub(super) program: &'b Program,
    pub(super) inputs: Inputs,
    pub(super) crt: CrtRandom,
    pub(super) mode: i32,
    pub(super) sentinels: Vec<u32>,
    pub(super) assertions: u64,
    features: BTreeMap<u32, Json>,
}

fn at(arguments: &[Value], index: usize) -> Result<&Value> {
    arguments
        .get(index)
        .ok_or_else(|| invalid("source helper argument missing"))
}
fn number(arguments: &[Value], index: usize) -> Result<Number> {
    at(arguments, index)?.number().map_err(invalid)
}
fn tensor(value: &Value) -> Result<u32> {
    if let Value::Handle(Handle::Tensor(id)) = value {
        Ok(*id)
    } else {
        Err(invalid("source helper expects tensor handle"))
    }
}
fn context(value: &Value) -> Result<u32> {
    if let Value::Handle(Handle::Context(id)) = value {
        Ok(*id)
    } else {
        Err(invalid("source helper expects context handle"))
    }
}
fn backend(value: &Value) -> Result<u32> {
    match value {
        Value::Handle(Handle::Backend(id))
        | Value::BackendDevice(id)
        | Value::BackendRegistry(id) => Ok(*id),
        _ => Err(invalid("source helper expects backend provenance")),
    }
}
fn string(value: &Value) -> Result<String> {
    match value {
        Value::Bytes(bytes) => String::from_utf8(bytes.clone()).map_err(Into::into),
        Value::Pointer { value, index } => {
            let text = string(value)?;
            let bytes = text
                .as_bytes()
                .get(*index..)
                .ok_or_else(|| invalid("source string pointer exceeds extent"))?;
            let end = bytes
                .iter()
                .position(|&byte| byte == 0)
                .unwrap_or(bytes.len());
            String::from_utf8(bytes[..end].to_vec()).map_err(Into::into)
        }
        _ => Err(invalid("source helper expects string")),
    }
}
fn typed_array(element: Element, values: impl IntoIterator<Item = Number>) -> Result<Value> {
    let values: Vec<_> = values.into_iter().collect();
    let mut buffer = Buffer::new(element, values.len()).map_err(invalid)?;
    for (index, value) in values.into_iter().enumerate() {
        buffer.set(index, value).map_err(invalid)?;
    }
    Ok(Value::Buffer(Rc::new(RefCell::new(buffer))))
}

fn write_bytes(destination: &Value, bytes: &[u8]) -> Result<()> {
    match destination {
        Value::Pointer { value, index } => {
            let Value::Buffer(buffer) = value.as_ref() else {
                return Err(invalid(
                    "source byte destination pointer lacks typed buffer",
                ));
            };
            let mut buffer = buffer.borrow_mut();
            let start = index
                .checked_mul(buffer.element.bytes())
                .ok_or_else(|| invalid("source byte write offset overflow"))?;
            buffer
                .bytes
                .get_mut(start..start + bytes.len())
                .ok_or_else(|| invalid("source byte write exceeds buffer"))?
                .copy_from_slice(bytes);
            Ok(())
        }
        Value::Buffer(buffer) => {
            buffer
                .borrow_mut()
                .bytes
                .get_mut(..bytes.len())
                .ok_or_else(|| invalid("source byte write exceeds buffer"))?
                .copy_from_slice(bytes);
            Ok(())
        }
        Value::Reference(place) | Value::Address(place) => {
            let value = place.get().map_err(invalid)?;
            if let Value::Number(number) = value {
                let replacement = match number.kind() {
                    Kind::F32 => Number::F32(f32::from_le_bytes(bytes.try_into()?)),
                    Kind::F64 => Number::F64(f64::from_le_bytes(bytes.try_into()?)),
                    Kind::I32 => Number::I32(i32::from_le_bytes(bytes.try_into()?)),
                    Kind::U32 => Number::U32(u32::from_le_bytes(bytes.try_into()?)),
                    Kind::I64 => Number::I64(i64::from_le_bytes(bytes.try_into()?)),
                    Kind::U64 => Number::U64(u64::from_le_bytes(bytes.try_into()?)),
                };
                place.set(Value::Number(replacement)).map_err(invalid)
            } else {
                write_bytes(&value, bytes)
            }
        }
        _ => Err(invalid("source byte destination is not writable storage")),
    }
}

impl<'a, 'b> Production<'a, 'b> {
    pub(super) const fn new(
        graph: Plan<'a>,
        program: &'b Program,
        seed: u64,
        threads: usize,
        mode: i32,
    ) -> Self {
        Self {
            graph,
            program,
            inputs: Inputs::new(seed, threads),
            crt: CrtRandom::new(1),
            mode,
            sentinels: Vec::new(),
            assertions: 0,
            features: BTreeMap::new(),
        }
    }

    fn feature_info(&mut self, id: u32) -> Result<Json> {
        if !self.features.contains_key(&id) {
            self.graph
                .push(json!({"Op":"backend-features","Backend":id}));
            let values = self.graph.flush()?;
            let observed = values
                .last()
                .ok_or_else(|| invalid("production backend features observation missing"))?
                .clone();
            self.features.insert(id, observed);
        }
        self.features
            .get(&id)
            .cloned()
            .ok_or_else(|| invalid("production backend features not retained"))
    }

    pub(super) fn add_sentinel(&mut self, context: u32) -> Result<()> {
        if self.mode != 0 {
            return Ok(());
        }
        let kind = self.program.constants["GGML_TYPE_F32"]
            .as_i64()
            .ok_or_else(|| invalid("source sentinel type constant differs"))?;
        let value = self.graph.call(
            "ggml_new_tensor_1d",
            &[
                Value::Handle(Handle::Context(context)),
                Value::Number(Number::I32(kind as i32)),
                Value::Number(Number::I64(1024)),
            ],
        )?;
        let id = tensor(&value)?;
        let name = format!("sent_{}", self.sentinels.len());
        let _ = self
            .graph
            .call("ggml_set_name", &[value, Value::Bytes(name.into_bytes())])?;
        self.sentinels.push(id);
        Ok(())
    }

    fn traverse(&mut self, name: &str, arguments: &[Value]) -> Result<Value> {
        let context = context(at(arguments, 0)?)?;
        if !self.graph.contexts_current {
            self.graph.snapshot_contexts()?;
        }
        let index = if name == "ggml_get_first_tensor" {
            0
        } else {
            let current = self.graph.tensor(tensor(at(arguments, 1)?)?)?.id;
            let order = self
                .graph
                .contexts
                .get(&context)
                .ok_or_else(|| invalid("source context traversal missing"))?;
            order
                .iter()
                .position(|&id| id == current)
                .ok_or_else(|| invalid("source next_tensor current not in context"))?
                + 1
        };
        let order = self
            .graph
            .contexts
            .get(&context)
            .ok_or_else(|| invalid("source context traversal missing"))?;
        Ok(order
            .get(index)
            .copied()
            .map_or(Value::Null, |id| Value::Handle(Handle::Tensor(id))))
    }

    fn tensor_transfer(&mut self, name: &str, arguments: &[Value]) -> Result<Value> {
        let id = tensor(at(arguments, 0)?)?;
        let offset = number(arguments, 2)?.as_u64();
        let count = number(arguments, 3)?.as_u64();
        if name == "ggml_backend_tensor_set" {
            let bytes = at(arguments, 1)?
                .bytes(usize::try_from(count)?)
                .map_err(invalid)?;
            self.inputs
                .set(&mut self.graph, id, &bytes, offset, "class-initializer")?;
        } else {
            let path = self.inputs.name(&self.graph, "source-readback")?;
            self.graph.push(json!({"Op":"tensor-get","Tensor":id,"File":path,"FileOffset":0,"TensorOffset":offset,"Bytes":count}));
            let _ = self.graph.flush()?;
            let path = self.graph.session.checked_file(&path)?;
            if fs::metadata(&path)?.len() != count {
                return Err(invalid("source readback byte extent differs"));
            }
            let bytes = fs::read(path)?;
            write_bytes(at(arguments, 1)?, &bytes)?;
        }
        Ok(Value::Null)
    }

    fn memory_copy(&mut self, arguments: &[Value]) -> Result<Value> {
        let count = usize::try_from(number(arguments, 2)?.as_u64())?;
        let bytes = at(arguments, 1)?.bytes(count).map_err(invalid)?;
        let destination = at(arguments, 0)?;
        if let Value::TensorBytes { tensor, field } = destination {
            if field != "op_params" {
                return Err(invalid("source raw tensor byte write has unknown field"));
            }
            let metadata = self.graph.tensor(*tensor)?;
            let mut params = decode_hex(&metadata.op_params_hex)?;
            params
                .get_mut(..count)
                .ok_or_else(|| invalid("source op_params copy exceeds metadata"))?
                .copy_from_slice(&bytes);
            self.graph.push(json!({"Op":"tensor-write","Tensor":tensor,"Fields":{"OpParamsHex":encode_hex(&params)}}));
            self.graph.invalidate_tensors();
        } else {
            write_bytes(destination, &bytes)?;
        }
        Ok(destination.clone())
    }
}

pub(super) fn encode_hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut text = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        text.push(char::from(DIGITS[usize::from(byte >> 4)]));
        text.push(char::from(DIGITS[usize::from(byte & 15)]));
    }
    text
}
fn decode_hex(text: &str) -> Result<Vec<u8>> {
    if !text.len().is_multiple_of(2) {
        return Err(invalid("production hex byte count differs"));
    }
    text.as_bytes()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| {
            let high = char::from(pair[0])
                .to_digit(16)
                .ok_or_else(|| invalid("production hex digit differs"))?;
            let low = char::from(pair[1])
                .to_digit(16)
                .ok_or_else(|| invalid("production hex digit differs"))?;
            Ok((high * 16 + low) as u8)
        })
        .collect()
}

impl Host for Production<'_, '_> {
    #[allow(
        clippy::too_many_lines,
        reason = "Source helper dispatch explicitly separates Rust-owned policy from raw production calls."
    )]
    fn call(&mut self, name: &str, arguments: &[Value]) -> Result<Option<Value>> {
        let result = match name {
            "source-assert" => {
                self.assertions += 1;
                Value::Null
            }
            "pointer-equal" => Value::Bool(
                self.graph.tensor(tensor(at(arguments, 0)?)?)?.id
                    == self.graph.tensor(tensor(at(arguments, 1)?)?)?.id,
            ),
            "random-device" => Value::Number(Number::U32(self.inputs.ledger.next())),
            "rand" => Value::Number(Number::I32(self.crt.next())),
            "ggml_get_first_tensor" | "ggml_get_next_tensor" => self.traverse(name, arguments)?,
            "ggml_backend_get_device" => Value::BackendDevice(backend(at(arguments, 0)?)?),
            "ggml_backend_dev_backend_reg" => Value::BackendRegistry(backend(at(arguments, 0)?)?),
            "ggml_backend_reg_name" => {
                let info = self.feature_info(backend(at(arguments, 0)?)?)?;
                Value::Bytes(
                    info["RegistryName"]
                        .as_str()
                        .ok_or_else(|| invalid("production registry name missing"))?
                        .as_bytes()
                        .to_vec(),
                )
            }
            "backend_has_feature" => {
                let info = self.feature_info(backend(at(arguments, 0)?)?)?;
                let requested = string(at(arguments, 1)?)?;
                Value::Bool(info["Features"].as_array().is_some_and(|features| {
                    features.iter().any(|feature| {
                        feature["Name"].as_str() == Some(requested.as_str())
                            && feature["Value"].as_str() == Some("1")
                    })
                }))
            }
            "init_tensor_uniform" | "init_tensor_kq_mask" | "init_tensor_tril" => {
                let id = tensor(at(arguments, 0)?)?;
                let minimum = arguments
                    .get(1)
                    .map(Value::number)
                    .transpose()
                    .map_err(invalid)?
                    .map_or(-1.0, Number::as_f32);
                let maximum = arguments
                    .get(2)
                    .map(Value::number)
                    .transpose()
                    .map_err(invalid)?
                    .map_or(1.0, Number::as_f32);
                match name {
                    "init_tensor_uniform" => {
                        self.inputs
                            .uniform(&mut self.graph, self.program, id, minimum, maximum)?;
                    }
                    "init_tensor_kq_mask" => {
                        self.inputs
                            .mask(&mut self.graph, self.program, id, minimum, maximum)?;
                    }
                    _ => self.inputs.triangular(
                        &mut self.graph,
                        self.program,
                        id,
                        minimum,
                        maximum,
                    )?,
                }
                Value::Null
            }
            "initialize_tensors" | "test_case::initialize_tensors" => {
                let ctx = context(at(arguments, 0)?)?;
                self.graph.snapshot_contexts()?;
                let order = self
                    .graph
                    .contexts
                    .get(&ctx)
                    .cloned()
                    .ok_or_else(|| invalid("source base initializer context missing"))?;
                for id in order {
                    self.inputs
                        .uniform(&mut self.graph, self.program, id, -1.0, 1.0)?;
                }
                Value::Null
            }
            "ggml_backend_tensor_set" | "ggml_backend_tensor_get" => {
                self.tensor_transfer(name, arguments)?
            }
            "memcpy" => self.memory_copy(arguments)?,
            "ggml_is_view_op" => {
                let op = number(arguments, 0)?.as_i64();
                Value::Bool(
                    [
                        "GGML_OP_VIEW",
                        "GGML_OP_RESHAPE",
                        "GGML_OP_PERMUTE",
                        "GGML_OP_TRANSPOSE",
                    ]
                    .iter()
                    .any(|name| self.program.constants[*name].as_i64() == Some(op)),
                )
            }
            "max_nmse_err" | "test_case::max_nmse_err" => Value::Number(Number::F64(1e-7)),
            "max_maa_err" => Value::Number(Number::F64(1e-4)),
            "grad_eps" => Value::Number(Number::F32(0.1)),
            "grad_precise" | "run_whole_graph" | "use_weight_context" => Value::Bool(false),
            "grad_nmax" => Value::Number(Number::I64(10000)),
            "ctx_tensors" => Value::Number(Number::U64(128)),
            "op_flops" => Value::Number(Number::U64(0)),
            "vars" => Value::Bytes(Vec::new()),
            "grad_expect" | "fusion_test_nodes" => Value::array(Vec::new()),
            "op_size" => {
                let value = self.graph.tensor(tensor(at(arguments, 0)?)?)?;
                let mut bytes = value.bytes;
                for id in value.sources.into_iter().flatten() {
                    bytes += self.graph.tensor(id)?.bytes;
                }
                Value::Number(Number::U64(bytes))
            }
            "nmse" | "test_case::err" => {
                let count = usize::try_from(number(arguments, 2)?.as_u64())?;
                let floats = |value: &Value| -> Result<Vec<f32>> {
                    (0..count)
                        .map(|index| {
                            Ok(value
                                .index(index)
                                .map_err(invalid)?
                                .number()
                                .map_err(invalid)?
                                .as_f32())
                        })
                        .collect()
                };
                Value::Number(Number::F64(metrics::nmse(
                    &floats(at(arguments, 0)?)?,
                    &floats(at(arguments, 1)?)?,
                )))
            }
            name if name.starts_with("ggml_") => {
                let result = self.graph.call(name, arguments)?;
                if name.starts_with("ggml_new_tensor") {
                    let ctx = context(at(arguments, 0)?)?;
                    self.add_sentinel(ctx)?;
                }
                result
            }
            _ => return Ok(None),
        };
        Ok(Some(result))
    }

    fn member(&mut self, value: &Value, name: &str) -> Result<Option<Value>> {
        let Value::Handle(Handle::Tensor(id)) = value else {
            return Ok(None);
        };
        let tensor = self.graph.tensor(*id)?;
        let value = match name {
            "ne" => typed_array(Element::I64, tensor.ne.map(Number::I64))?,
            "nb" => typed_array(Element::U64, tensor.nb.map(Number::U64))?,
            "src" => Value::array(
                tensor
                    .sources
                    .into_iter()
                    .map(|id| id.map_or(Value::Null, |id| Value::Handle(Handle::Tensor(id))))
                    .collect(),
            ),
            "type" => Value::Number(Number::I32(tensor.kind)),
            "op" => Value::Number(Number::I32(tensor.op)),
            "name" => Value::Bytes(tensor.name.into_bytes()),
            "op_params" => Value::TensorBytes {
                tensor: *id,
                field: name.into(),
            },
            _ => return Ok(None),
        };
        Ok(Some(value))
    }

    fn write(
        &mut self,
        value: &Value,
        name: &str,
        index: Option<usize>,
        replacement: Value,
    ) -> Result<()> {
        let id = tensor(value)?;
        let tensor = self.graph.tensor(id)?;
        let fields = match (name, index) {
            ("op", None) => json!({"Op":replacement.number().map_err(invalid)?.as_i64()}),
            ("nb", Some(index)) => {
                let mut strides = tensor.nb;
                *strides
                    .get_mut(index)
                    .ok_or_else(|| invalid("source tensor stride index exceeds dimensions"))? =
                    replacement.number().map_err(invalid)?.as_u64();
                json!({"NB":strides})
            }
            ("src", Some(index)) => {
                let mut sources = tensor.sources;
                *sources
                    .get_mut(index)
                    .ok_or_else(|| invalid("source tensor source index exceeds slots"))? =
                    if matches!(replacement, Value::Null) {
                        None
                    } else {
                        Some(self::tensor(&replacement)?)
                    };
                json!({"Sources":sources})
            }
            _ => return Err(invalid("unsupported source metadata assignment")),
        };
        self.graph
            .push(json!({"Op":"tensor-write","Tensor":id,"Fields":fields}));
        self.graph.invalidate_tensors();
        Ok(())
    }

    fn force(&mut self, value: Value) -> Result<Value> {
        if let Value::NativeInteger { kind, argument } = value {
            let value = self.graph.resolve_integer(&argument)?;
            if let Value::Number(number) = value {
                Ok(Value::Number(number.cast(kind)))
            } else {
                Ok(value)
            }
        } else {
            Ok(value)
        }
    }
}
