// SPDX-License-Identifier: MIT
//! Typed emission of production calls from Rust-owned source expressions.

use super::super::{Result, invalid};
use super::number::{Kind, Number};
use super::program::{Parameter, SOURCE_SHA256};
use super::tensor::Tensor;
use super::transport::Session;
use super::value::{Handle, Value};
use serde::Deserialize;
use serde_json::{Value as Json, json};
use std::collections::BTreeMap;
use std::sync::OnceLock;

#[derive(Deserialize)]
#[serde(rename_all = "PascalCase")]
struct Intrinsic {
    #[serde(rename = "Return")]
    result: String,
    parameters: Vec<Parameter>,
}

#[derive(Deserialize)]
#[serde(rename_all = "PascalCase")]
struct Library {
    schema_version: u32,
    source_sha256: String,
    upstream_commit: String,
    intrinsics: BTreeMap<String, Intrinsic>,
}

fn library() -> Result<&'static Library> {
    static LIBRARY: OnceLock<std::result::Result<Library, String>> = OnceLock::new();
    match LIBRARY.get_or_init(|| {
        let library: Library = serde_json::from_str(include_str!("../backend-ops-intrinsics.json"))
            .map_err(|error| format!("backend production signatures: {error}"))?;
        if library.schema_version != 1
            || library.source_sha256 != SOURCE_SHA256
            || library.upstream_commit != "b87ca7225456b06b0975b8239c66b5aecf07e37c"
            || library.intrinsics.len() != 173
        {
            return Err("backend production signature provenance or extent differs".into());
        }
        Ok(library)
    }) {
        Ok(library) => Ok(library),
        Err(error) => Err(invalid(error.clone())),
    }
}

pub(super) fn return_type(name: &str) -> Result<&'static str> {
    library()?
        .intrinsics
        .get(name)
        .map(|intrinsic| intrinsic.result.as_str())
        .ok_or_else(|| invalid("Rust backend call is absent from pinned signatures"))
}

fn scalar(value: &Json) -> Result<Value> {
    let value = value
        .as_object()
        .filter(|object| object.len() == 1)
        .ok_or_else(|| invalid("backend scalar observation must have one type tag"))?;
    let (kind, value) = value
        .iter()
        .next()
        .ok_or_else(|| invalid("backend scalar observation is empty"))?;
    let number = match kind.as_str() {
        "String" => {
            return if value.is_null() {
                Ok(Value::Null)
            } else {
                value
                    .as_str()
                    .map(|value| Value::Pointer {
                        value: Box::new(Value::Bytes(value.as_bytes().to_vec())),
                        index: 0,
                    })
                    .ok_or_else(|| invalid("backend String observation differs"))
            };
        }
        "Bool" => {
            return value
                .as_bool()
                .map(Value::Bool)
                .ok_or_else(|| invalid("backend Bool observation differs"));
        }
        "I32" => Number::I32(i32::try_from(
            value
                .as_i64()
                .ok_or_else(|| invalid("backend I32 observation differs"))?,
        )?),
        "I64" => Number::I64(
            value
                .as_i64()
                .ok_or_else(|| invalid("backend I64 observation differs"))?,
        ),
        "U64" => Number::U64(
            value
                .as_u64()
                .ok_or_else(|| invalid("backend U64 observation differs"))?,
        ),
        "F32Bits" => Number::F32(f32::from_bits(u32::try_from(
            value
                .as_u64()
                .ok_or_else(|| invalid("backend F32 observation differs"))?,
        )?)),
        "F64Bits" => Number::F64(f64::from_bits(
            value
                .as_u64()
                .ok_or_else(|| invalid("backend F64 observation differs"))?,
        )),
        _ => return Err(invalid("unknown backend scalar observation type")),
    };
    Ok(Value::Number(number))
}

fn pointer_values(value: &Value) -> Result<Vec<Value>> {
    let (value, start) = match value {
        Value::Pointer { value, index } => (value.as_ref(), *index),
        Value::Reference(place) => return pointer_values(&place.get().map_err(invalid)?),
        value => (value, 0),
    };
    let length = value.len().map_err(invalid)?;
    (start..length)
        .map(|index| value.index(index).map_err(invalid))
        .collect()
}

fn pointer_argument(value: &Value, kind: &str) -> Result<Json> {
    let clean = kind.replace("const", "").replace("struct", "");
    if clean == "ggml_tensor**" {
        let values = pointer_values(value)?;
        let tensors = values
            .iter()
            .map(|value| match value {
                Value::Handle(Handle::Tensor(id)) => Ok(json!(id)),
                Value::Null => Ok(Json::Null),
                _ => Err(invalid("production tensor array contains a non-tensor")),
            })
            .collect::<Result<Vec<_>>>()?;
        return Ok(json!({"TensorArray":tensors}));
    }
    if clean == "ggml_tensor*" || clean == "ggml_context*" || clean == "ggml_cgraph*" {
        let Value::Handle(handle) = value else {
            return Err(invalid("production pointer argument is not a handle"));
        };
        if !matches!(
            (clean.as_str(), handle),
            ("ggml_tensor*", Handle::Tensor(_))
                | ("ggml_context*", Handle::Context(_))
                | ("ggml_cgraph*", Handle::Graph(_))
        ) {
            return Err(invalid("production pointer handle type differs"));
        }
        return Ok(handle.wire());
    }
    if clean == "char*" {
        let (bytes, start) = match value {
            Value::Bytes(bytes) => (bytes.as_slice(), 0),
            Value::Pointer { value, index } => {
                let Value::Bytes(bytes) = value.as_ref() else {
                    return Err(invalid("production string pointer has no byte string"));
                };
                (bytes.as_slice(), *index)
            }
            _ => return Err(invalid("production string argument has no bytes")),
        };
        let bytes = bytes
            .get(start..)
            .ok_or_else(|| invalid("production string pointer is past end"))?;
        let end = bytes
            .iter()
            .position(|&byte| byte == 0)
            .unwrap_or(bytes.len());
        let text = std::str::from_utf8(&bytes[..end])?;
        return Ok(json!({"String":text}));
    }
    let values = pointer_values(value)?;
    if clean == "int64_t*" || clean == "int*" || clean == "int32_t*" {
        let values = values
            .iter()
            .map(|value| match value {
                Value::NativeInteger { argument, .. } => Ok(argument.clone()),
                _ => Ok(json!(value.number().map_err(invalid)?.as_i64())),
            })
            .collect::<Result<Vec<_>>>()?;
        return Ok(if clean == "int64_t*" {
            json!({"I64Array":values})
        } else {
            json!({"I32Array":values})
        });
    }
    if clean == "float*" {
        let values = values
            .iter()
            .map(|value| Ok(value.number().map_err(invalid)?.as_f32().to_bits()))
            .collect::<Result<Vec<_>>>()?;
        return Ok(json!({"F32BitsArray":values}));
    }
    Err(invalid(format!("unhandled production pointer type {kind}")))
}

fn argument(value: &Value, kind: &str) -> Result<Json> {
    if matches!(value, Value::Null) {
        if !kind.contains('*') && kind != "ggml_backend_t" && kind != "ggml_backend_buffer_t" {
            return Err(invalid(
                "null supplied for a non-pointer production argument",
            ));
        }
        return Ok(Json::Null);
    }
    if let Value::Reference(place) = value {
        return argument(&place.get().map_err(invalid)?, kind);
    }
    if kind.contains('*') {
        return pointer_argument(value, kind);
    }
    if kind == "ggml_backend_t" || kind == "ggml_backend_buffer_t" {
        return match value {
            Value::Handle(Handle::Backend(id)) if kind == "ggml_backend_t" => {
                Ok(json!({"Backend":id}))
            }
            Value::Handle(Handle::Buffer(id)) if kind == "ggml_backend_buffer_t" => {
                Ok(json!({"Buffer":id}))
            }
            _ => Err(invalid("production backend/buffer handle type differs")),
        };
    }
    if let Value::NativeInteger { argument, .. } = value {
        return Ok(argument.clone());
    }
    if kind == "bool" {
        return Ok(json!({"Bool":value.truth().map_err(invalid)?}));
    }
    Ok(value
        .number()
        .map_err(invalid)?
        .cast(Kind::from_type(kind))
        .wire())
}

pub(super) struct Plan<'a> {
    pub(super) session: &'a mut Session,
    pending: Vec<Json>,
    pub(super) tensors: BTreeMap<u32, Tensor>,
    pub(super) contexts: BTreeMap<u32, Vec<u32>>,
    pub(super) contexts_current: bool,
    pub(super) graphs: BTreeMap<u32, Json>,
    scalars: BTreeMap<u32, Value>,
    shape_queries: BTreeMap<(String, u32), Value>,
    next_context: u32,
    next_backend: u32,
    next_tensor: u32,
    next_graph: u32,
    next_buffer: u32,
    next_scalar: u32,
}

impl<'a> Plan<'a> {
    pub(super) const fn new(session: &'a mut Session) -> Self {
        Self {
            session,
            pending: Vec::new(),
            tensors: BTreeMap::new(),
            contexts: BTreeMap::new(),
            contexts_current: false,
            graphs: BTreeMap::new(),
            scalars: BTreeMap::new(),
            shape_queries: BTreeMap::new(),
            next_context: 0,
            next_backend: 0,
            next_tensor: 0,
            next_graph: 0,
            next_buffer: 0,
            next_scalar: 0,
        }
    }

    pub(super) fn push(&mut self, action: Json) {
        if matches!(
            action["Op"].as_str(),
            Some("tensor-write" | "context-init" | "context-free")
        ) {
            self.invalidate_tensors();
            self.shape_queries.clear();
        }
        self.pending.push(action);
    }

    pub(super) fn flush(&mut self) -> Result<Vec<Json>> {
        if self.pending.is_empty() {
            return Ok(Vec::new());
        }
        let actions = std::mem::take(&mut self.pending);
        let observed = self.session.exchange(&actions)?;
        for (action, value) in actions.iter().zip(&observed) {
            match action["Op"].as_str() {
                Some("context-snapshot") => {
                    let context = u32::try_from(
                        action["Context"]
                            .as_u64()
                            .ok_or_else(|| invalid("backend context handle differs"))?,
                    )?;
                    let mut order = Vec::new();
                    for tensor in value["Tensors"]
                        .as_array()
                        .ok_or_else(|| invalid("backend context snapshot differs"))?
                    {
                        let tensor: Tensor = serde_json::from_value(tensor.clone())?;
                        order.push(tensor.id);
                        for &alias in &tensor.aliases {
                            self.tensors.insert(alias, tensor.clone());
                        }
                        self.tensors.insert(tensor.id, tensor);
                    }
                    self.contexts.insert(context, order);
                }
                Some("graph-snapshot") => {
                    let graph = u32::try_from(
                        action["Graph"]
                            .as_u64()
                            .ok_or_else(|| invalid("backend graph handle differs"))?,
                    )?;
                    self.graphs.insert(graph, value.clone());
                }
                Some("tensor-snapshot") => {
                    let tensor: Tensor = serde_json::from_value(value["Tensor"].clone())?;
                    for &alias in &tensor.aliases {
                        self.tensors.insert(alias, tensor.clone());
                    }
                    self.tensors.insert(tensor.id, tensor);
                }
                Some("scalar") => {
                    let id = u32::try_from(
                        action["Scalar"]
                            .as_u64()
                            .ok_or_else(|| invalid("backend scalar handle differs"))?,
                    )?;
                    self.scalars.insert(id, scalar(&value["Value"])?);
                }
                _ => {}
            }
        }
        Ok(observed)
    }

    pub(super) fn context(&mut self, memory: u64) -> Handle {
        let id = self.next_context;
        self.next_context += 1;
        self.contexts.insert(id, Vec::new());
        self.push(json!({"Op":"context-init","Id":id,"MemoryBytes":memory,"NoAlloc":true}));
        Handle::Context(id)
    }

    pub(super) fn backend(&mut self, threads: Option<usize>, reference: bool) -> Handle {
        let id = self.next_backend;
        self.next_backend += 1;
        self.push(json!({"Op":"backend-init","Id":id,"Device":"CPU","Threads":threads,"UseRef":reference}));
        Handle::Backend(id)
    }

    pub(super) fn allocate(&mut self, context: u32, backend: u32) -> Handle {
        let id = self.next_buffer;
        self.next_buffer += 1;
        self.push(
            json!({"Op":"allocate","Context":context,"Backend":backend,"Result":{"Buffer":id}}),
        );
        Handle::Buffer(id)
    }

    pub(super) fn call(&mut self, name: &str, values: &[Value]) -> Result<Value> {
        let intrinsic = library()?
            .intrinsics
            .get(name)
            .ok_or_else(|| invalid(format!("source production intrinsic {name} is missing")))?;
        if intrinsic.parameters.len() != values.len() {
            return Err(invalid(format!(
                "source production argument count differs for {name}"
            )));
        }
        // These immutable shape queries occur inside original per-element
        // initializer loops. Retain the actual typed production observation
        // until a metadata mutation, rather than emitting the same native
        // query hundreds of thousands of times and exhausting the action cap.
        let shape_key = if matches!(name, "ggml_nelements" | "ggml_nrows") {
            match values.first() {
                Some(Value::Handle(Handle::Tensor(id))) => Some((name.to_owned(), *id)),
                _ => None,
            }
        } else {
            None
        };
        if let Some(value) = shape_key
            .as_ref()
            .and_then(|key| self.shape_queries.get(key))
        {
            return Ok(value.clone());
        }
        let arguments = values
            .iter()
            .zip(&intrinsic.parameters)
            .map(|(value, parameter)| argument(value, &parameter.kind))
            .collect::<Result<Vec<_>>>()?;
        let (result, value) = match intrinsic.result.as_str() {
            "void" => (Json::Null, Value::Null),
            "structggml_tensor*" => {
                let id = self.next_tensor;
                self.next_tensor += 1;
                (json!({"Tensor":id}), Value::Handle(Handle::Tensor(id)))
            }
            "structggml_cgraph*" => {
                let id = self.next_graph;
                self.next_graph += 1;
                (json!({"Graph":id}), Value::Handle(Handle::Graph(id)))
            }
            kind => {
                let id = self.next_scalar;
                self.next_scalar += 1;
                (
                    json!({"Scalar":id}),
                    Value::NativeInteger {
                        kind: Kind::from_type(kind),
                        argument: json!({"Scalar":id}),
                    },
                )
            }
        };
        if intrinsic.result == "void" || matches!(value, Value::Handle(Handle::Tensor(_))) {
            self.invalidate_tensors();
            self.shape_queries.clear();
        }
        self.push(json!({"Op":"call","Function":name,"Arguments":arguments,"Result":result}));
        if let Some(key) = shape_key {
            let id = u32::try_from(
                result["Scalar"]
                    .as_u64()
                    .ok_or_else(|| invalid("shape query scalar is missing"))?,
            )?;
            let observed = self.observe_scalar(id)?;
            self.shape_queries.insert(key, observed.clone());
            return Ok(observed);
        }
        Ok(value)
    }

    pub(super) fn snapshot_contexts(&mut self) -> Result<()> {
        self.pending.extend(
            self.contexts
                .keys()
                .map(|id| json!({"Op":"context-snapshot","Context":id})),
        );
        let _ = self.flush()?;
        self.contexts_current = true;
        Ok(())
    }

    pub(super) fn invalidate_tensors(&mut self) {
        self.tensors.clear();
        self.contexts_current = false;
    }

    pub(super) fn tensor(&mut self, id: u32) -> Result<Tensor> {
        if !self.tensors.contains_key(&id) {
            self.push(json!({"Op":"tensor-snapshot","Tensor":id}));
            let _ = self.flush()?;
        }
        self.tensors
            .get(&id)
            .cloned()
            .ok_or_else(|| invalid("source tensor handle is absent from production contexts"))
    }

    pub(super) fn observe_scalar(&mut self, id: u32) -> Result<Value> {
        if !self.scalars.contains_key(&id) {
            self.push(json!({"Op":"scalar","Scalar":id,"Release":true}));
            let _ = self.flush()?;
        }
        self.scalars
            .get(&id)
            .cloned()
            .ok_or_else(|| invalid("source scalar handle observation is missing"))
    }

    pub(super) fn resolve_integer(&mut self, value: &Json) -> Result<Value> {
        if let Some(id) = value.get("Scalar").and_then(Json::as_u64) {
            return self.observe_scalar(u32::try_from(id)?);
        }
        if let Some(field) = value.get("TensorField") {
            let tensor = self.tensor(u32::try_from(
                field["Tensor"]
                    .as_u64()
                    .ok_or_else(|| invalid("source tensor field handle differs"))?,
            )?)?;
            let name = field["Field"]
                .as_str()
                .ok_or_else(|| invalid("source tensor field name differs"))?;
            let value = match name {
                "ne" | "nb" => {
                    let index = usize::try_from(
                        field["Index"]
                            .as_u64()
                            .ok_or_else(|| invalid("source tensor field index differs"))?,
                    )?;
                    if index >= 4 {
                        return Err(invalid("source tensor field index exceeds dimensions"));
                    }
                    if name == "ne" {
                        Number::I64(tensor.ne[index])
                    } else {
                        Number::U64(tensor.nb[index])
                    }
                }
                "type" => Number::I32(tensor.kind),
                "op" => Number::I32(tensor.op),
                "flags" => Number::I32(tensor.flags),
                _ => return Err(invalid("source tensor scalar field is unknown")),
            };
            return Ok(Value::Number(value));
        }
        if let Some(expression) = value.get("Arithmetic") {
            let operands = expression["Arguments"]
                .as_array()
                .filter(|values| values.len() == 2)
                .ok_or_else(|| invalid("source metadata arithmetic arity differs"))?;
            let a = self
                .resolve_integer(&operands[0])?
                .number()
                .map_err(invalid)?;
            let b = self
                .resolve_integer(&operands[1])?
                .number()
                .map_err(invalid)?;
            let op = expression["Op"]
                .as_str()
                .ok_or_else(|| invalid("source metadata arithmetic operator differs"))?;
            return Ok(Value::Number(a.binary(op, b).map_err(invalid)?));
        }
        scalar(value)
    }
}
