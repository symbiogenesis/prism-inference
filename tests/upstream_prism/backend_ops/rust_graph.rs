//! Continuation serialization for the production owned CPU backend.

use super::{
    Argument, CpuBackend, Observation, Result, TensorId, Value, array, hex, integer, invalid, json,
    text, unhex, unsigned,
};
use bonsai_prism_runtime::backend::{GraphId, operation_name, type_layout, unary_name};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

const MAX_ACTIONS: usize = 524_288;
const MAX_ACTION_BYTES: u64 = 128 * 1024 * 1024;
const MAX_OBSERVATION_BYTES: usize = 16 * 1024 * 1024;

/// The Rust target executes the same bounded production actions directly.
/// Its state belongs to one geometry, with no polling files or worker thread.
pub(in super::super) struct GraphSession {
    state: State,
    requests: BufWriter<File>,
    responses: BufWriter<File>,
    total_actions: usize,
    deadline: Instant,
    poisoned: bool,
}

#[derive(Serialize)]
struct Trace<'a> {
    #[serde(rename = "Trace")]
    values: &'a [Value],
}

fn encode_trace(values: &[Value]) -> Result<Vec<u8>> {
    let response = serde_json::to_vec(&Trace { values })?;
    if response.len() > MAX_OBSERVATION_BYTES {
        return Err(invalid("Rust backend observation exceeds frame bound"));
    }
    Ok(response)
}

impl GraphSession {
    pub(in super::super) fn open(
        directory: &Path,
        maximum: usize,
        deadline: Instant,
    ) -> Result<Self> {
        let state = State::open(directory, maximum)?;
        Ok(Self {
            requests: BufWriter::new(File::create(state.directory.join("actions.jsonl"))?),
            responses: BufWriter::new(File::create(state.directory.join("observations.jsonl"))?),
            state,
            total_actions: 1,
            deadline,
            poisoned: false,
        })
    }

    pub(in super::super) fn directory(&self) -> &Path {
        &self.state.directory
    }

    pub(in super::super) fn exchange(&mut self, actions: &[Value]) -> Result<Vec<Value>> {
        if self.poisoned {
            return Err(invalid(
                "Rust backend session is poisoned by an earlier failure",
            ));
        }
        let result = self.exchange_checked(actions);
        if result.is_err() {
            self.poisoned = true;
        }
        result
    }

    fn exchange_checked(&mut self, actions: &[Value]) -> Result<Vec<Value>> {
        // Match the file and native-pipe protocols' initial checkpoint and
        // per-exchange checkpoint slots even though no checkpoint is needed.
        let first_action = self.total_actions;
        self.total_actions = self
            .total_actions
            .checked_add(actions.len() + 1)
            .filter(|total| *total <= MAX_ACTIONS)
            .ok_or_else(|| invalid("Rust backend explicit action extent exceeds its bound"))?;
        if actions.is_empty() || Instant::now() >= self.deadline {
            return Err(invalid("Rust backend empty batch or expired case deadline"));
        }
        let request = serde_json::to_vec(actions)?;
        if request.len() as u64 > MAX_ACTION_BYTES {
            return Err(invalid("Rust backend request exceeds 128MiB"));
        }
        self.requests.write_all(&request)?;
        self.requests.write_all(b"\n")?;
        self.requests.flush()?;
        drop(request);
        let mut trace = Vec::with_capacity(actions.len());
        for (index, action) in actions.iter().enumerate() {
            if Instant::now() >= self.deadline {
                return Err(invalid("Rust backend case deadline expired"));
            }
            trace.push(self.state.run(action).map_err(|error| {
                invalid(format!(
                    "Rust backend action {}: {error}",
                    first_action + index
                ))
            })?);
        }
        let response = encode_trace(&trace)?;
        if Instant::now() >= self.deadline {
            return Err(invalid("Rust backend case deadline expired"));
        }
        self.responses.write_all(&response)?;
        self.responses.write_all(b"\n")?;
        self.responses.flush()?;
        Ok(trace)
    }
}

struct State {
    backend: CpuBackend,
    tensors: BTreeMap<u32, Option<TensorId>>,
    graphs: BTreeMap<u32, GraphId>,
    scalars: BTreeMap<u32, Value>,
    contexts: BTreeSet<u32>,
    backends: BTreeSet<u32>,
    buffers: BTreeSet<u32>,
    directory: PathBuf,
    maximum: usize,
    started: Instant,
}

fn id(value: &Value) -> Result<u32> {
    Ok(u32::try_from(unsigned(value)?)?)
}
fn lookup<T: Copy>(map: &BTreeMap<u32, T>, value: &Value) -> Result<T> {
    map.get(&id(value)?)
        .copied()
        .ok_or_else(|| invalid("Rust backend handle is unknown"))
}
type TensorAliases = BTreeMap<TensorId, Vec<u32>>;

fn tensor_aliases(tensors: &BTreeMap<u32, Option<TensorId>>) -> TensorAliases {
    let mut aliases = TensorAliases::new();
    for (&id, &tensor) in tensors {
        if let Some(tensor) = tensor {
            aliases.entry(tensor).or_default().push(id);
        }
    }
    aliases
}

fn canonical(aliases: &TensorAliases, tensor: TensorId) -> u32 {
    aliases
        .get(&tensor)
        .and_then(|ids| ids.first().copied())
        .unwrap_or_else(|| 0x8000_0000 + u32::try_from(tensor.0).unwrap_or(u32::MAX - 0x8000_0000))
}
fn metadata(backend: &CpuBackend, aliases: &TensorAliases, tensor: TensorId) -> Result<Value> {
    let info = backend.info(tensor)?;
    let layout = type_layout(info.kind)?;
    let sources = info
        .source_slots
        .iter()
        .map(|id| id.map_or(Value::Null, |id| json!(canonical(aliases, id))))
        .collect::<Vec<_>>();
    let description = if info.operation == 91 {
        unary_name(i32::from_le_bytes(info.parameters[..4].try_into()?))
    } else {
        operation_name(info.operation)
    };
    Ok(
        json!({"Id":canonical(aliases,tensor),"Aliases":aliases.get(&tensor).cloned().unwrap_or_default(),"Type":info.kind,"NE":info.shape,"NB":info.strides,"Op":info.operation,"Flags":info.flags,"Name":info.name,
        "ViewSource":info.view.map(|id|canonical(aliases,id)),"ViewOffset":info.offset,"Sources":sources,"OpParamsHex":hex(&info.parameters),"NBytes":info.bytes()?,"NElements":info.elements(),"NRows":info.rows(),"BlockSize":layout.block,"TypeSize":layout.bytes,"OpDescription":description,"Contiguous":info.contiguous(),"IsTransposed":info.strides[0]>info.strides[1]}),
    )
}

impl State {
    fn open(directory: &Path, maximum: usize) -> Result<Self> {
        let directory = fs::canonicalize(directory)?;
        if !directory.is_absolute() || !directory.is_dir() {
            return Err(invalid("Rust backend directory differs"));
        }
        Ok(Self {
            backend: CpuBackend::new(),
            tensors: BTreeMap::new(),
            graphs: BTreeMap::new(),
            scalars: BTreeMap::new(),
            contexts: BTreeSet::new(),
            backends: BTreeSet::new(),
            buffers: BTreeSet::new(),
            directory,
            maximum,
            started: Instant::now(),
        })
    }

    fn tensor(&self, value: &Value) -> Result<TensorId> {
        lookup(&self.tensors, value)?.ok_or_else(|| invalid("Rust backend tensor handle is null"))
    }
    fn file(&self, value: &Value) -> Result<PathBuf> {
        let path = Path::new(text(value)?);
        if !path.is_absolute() {
            return Err(invalid("Rust backend file path must be absolute"));
        }
        let path = if path.try_exists()? {
            fs::canonicalize(path)?
        } else {
            fs::canonicalize(
                path.parent()
                    .ok_or_else(|| invalid("Rust backend file has no parent"))?,
            )?
            .join(
                path.file_name()
                    .ok_or_else(|| invalid("Rust backend file has no name"))?,
            )
        };
        if !path.starts_with(&self.directory) || path == self.directory {
            return Err(invalid("Rust backend file is outside owned directory"));
        }
        Ok(path)
    }
    fn read(&self, path: &Path, offset: usize, bytes: usize) -> Result<Vec<u8>> {
        let metadata = fs::metadata(path)?;
        if !metadata.is_file()
            || metadata.len() > self.maximum as u64
            || offset as u64 > metadata.len()
            || bytes as u64 > metadata.len() - offset as u64
        {
            return Err(invalid("Rust backend bounded read differs"));
        }
        let mut file = File::open(path)?;
        file.seek(SeekFrom::Start(offset as u64))?;
        let mut data = Vec::new();
        data.try_reserve_exact(bytes)?;
        data.resize(bytes, 0);
        file.read_exact(&mut data)?;
        Ok(data)
    }
    fn write(&self, path: &Path, offset: usize, data: &[u8], truncate: bool) -> Result<()> {
        if offset > self.maximum || data.len() > self.maximum - offset {
            return Err(invalid("Rust backend file write exceeds bound"));
        }
        let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(truncate)
            .open(path)?;
        file.seek(SeekFrom::Start(offset as u64))?;
        file.write_all(data)?;
        file.flush()?;
        Ok(())
    }
    fn integer(&self, value: &Value, depth: usize) -> Result<i64> {
        if depth >= 24 {
            return Err(invalid("Rust backend integer nesting exceeds bound"));
        }
        for tag in ["I32", "I64", "U64"] {
            if let Some(number) = value.get(tag) {
                return integer(number);
            }
        }
        if let Some(scalar) = value.get("Scalar") {
            return self.integer(
                self.scalars
                    .get(&id(scalar)?)
                    .ok_or_else(|| invalid("Rust backend scalar missing"))?,
                depth + 1,
            );
        }
        if let Some(field) = value.get("TensorField") {
            let info = self.backend.info(self.tensor(&field["Tensor"])?)?;
            return Ok(match text(&field["Field"])? {
                "ne" => i64::try_from(
                    *info
                        .shape
                        .get(unsigned(&field["Index"])?)
                        .ok_or_else(|| invalid("Rust tensor dimension index differs"))?,
                )?,
                "nb" => i64::try_from(
                    *info
                        .strides
                        .get(unsigned(&field["Index"])?)
                        .ok_or_else(|| invalid("Rust tensor stride index differs"))?,
                )?,
                "type" => i64::from(info.kind),
                "op" => i64::from(info.operation),
                "flags" => i64::from(info.flags),
                _ => return Err(invalid("Rust tensor field is unknown")),
            });
        }
        if let Some(arithmetic) = value.get("Arithmetic") {
            let arguments = array(&arithmetic["Arguments"])?;
            if arguments.len() != 2 {
                return Err(invalid("Rust backend arithmetic must be binary"));
            }
            let left = self.integer(&arguments[0], depth + 1)?;
            let right = self.integer(&arguments[1], depth + 1)?;
            return match text(&arithmetic["Op"])? {
                "+" => left.checked_add(right),
                "-" => left.checked_sub(right),
                "*" => left.checked_mul(right),
                "/" => left.checked_div(right),
                "%" => left.checked_rem(right),
                _ => None,
            }
            .ok_or_else(|| {
                invalid("Rust backend integer arithmetic overflow or unknown operator")
            });
        }
        Err(invalid("Rust backend typed integer missing"))
    }
    fn argument(&self, value: &Value) -> Result<Argument> {
        if value.is_null() {
            return Ok(Argument::Null);
        }
        if let Some(scalar) = value.get("Scalar") {
            return self.argument(
                self.scalars
                    .get(&id(scalar)?)
                    .ok_or_else(|| invalid("Rust backend scalar missing"))?,
            );
        }
        if let Some(tensor) = value.get("Tensor") {
            return Ok(Argument::Tensor(self.tensor(tensor)?));
        }
        if let Some(graph) = value.get("Graph") {
            return Ok(Argument::Graph(lookup(&self.graphs, graph)?));
        }
        if let Some(context) = value.get("Context") {
            let context = id(context)?;
            if !self.contexts.contains(&context) {
                return Err(invalid("Rust backend context missing"));
            }
            return Ok(Argument::Context(context));
        }
        if let Some(bits) = value.get("F32Bits") {
            return Ok(Argument::Float(f32::from_bits(u32::try_from(unsigned(
                bits,
            )?)?)));
        }
        if let Some(boolean) = value.get("Bool") {
            return Ok(Argument::Bool(
                boolean
                    .as_bool()
                    .ok_or_else(|| invalid("Rust backend Bool missing"))?,
            ));
        }
        if let Some(string) = value.get("String") {
            return Ok(Argument::String(if string.is_null() {
                None
            } else {
                Some(text(string)?.to_owned())
            }));
        }
        for tag in ["I64Array", "I32Array"] {
            if let Some(values) = value.get(tag) {
                return Ok(Argument::Integers(
                    array(values)?
                        .iter()
                        .map(|value| {
                            if value.is_object() {
                                self.integer(value, 0)
                            } else {
                                integer(value)
                            }
                        })
                        .collect::<Result<Vec<_>>>()?,
                ));
            }
        }
        for tag in ["Backend", "Buffer"] {
            if let Some(handle) = value.get(tag) {
                let handle = id(handle)?;
                if !if tag == "Backend" {
                    self.backends.contains(&handle)
                } else {
                    self.buffers.contains(&handle)
                } {
                    return Err(invalid("Rust backend device/buffer missing"));
                }
                return Ok(Argument::Integer(i64::from(handle)));
            }
        }
        Ok(Argument::Integer(self.integer(value, 0)?))
    }
    fn call(&mut self, action: &Value) -> Result<Value> {
        let function = text(&action["Function"])?;
        let args = array(&action["Arguments"])?
            .iter()
            .map(|value| self.argument(value))
            .collect::<Result<Vec<_>>>()?;
        let observation = self.backend.call(function, &args)?;
        let result = &action["Result"];
        if result.is_null() {
            return Ok(json!({}));
        }
        match observation {
            Observation::Tensor(tensor) => {
                let handle = id(&result["Tensor"])?;
                if self.tensors.insert(handle, tensor).is_some() {
                    return Err(invalid("duplicate Rust tensor binding"));
                }
            }
            Observation::Graph(graph) => {
                let handle = id(&result["Graph"])?;
                if self.graphs.insert(handle, graph).is_some() {
                    return Err(invalid("duplicate Rust graph binding"));
                }
            }
            Observation::Integer(value) => {
                let signature = super::super::graph::return_type(function)?;
                let scalar = if signature == "size_t" {
                    json!({"U64":u64::try_from(value)?})
                } else if signature == "int64_t" {
                    json!({"I64":value})
                } else {
                    json!({"I32":i32::try_from(value)?})
                };
                self.scalars.insert(id(&result["Scalar"])?, scalar);
            }
            Observation::Bool(value) => {
                self.scalars
                    .insert(id(&result["Scalar"])?, json!({"Bool":value}));
            }
            Observation::String(value) => {
                self.scalars
                    .insert(id(&result["Scalar"])?, json!({"String":value}));
            }
            Observation::Void => return Err(invalid("Rust void graph call has a result binding")),
        }
        Ok(json!({}))
    }
    #[allow(
        clippy::too_many_lines,
        reason = "Each action only serializes a production backend observation."
    )]
    fn run(&mut self, action: &Value) -> Result<Value> {
        match text(&action["Op"])? {
            "context-init" => {
                let handle = id(&action["Id"])?;
                if self.contexts.len() >= 16 || !self.contexts.insert(handle) {
                    return Err(invalid("duplicate or excessive Rust context"));
                }
                Ok(json!({}))
            }
            "backend-init" => {
                let handle = id(&action["Id"])?;
                if text(&action["Device"])? != "CPU"
                    || self.backends.len() >= 4
                    || !self.backends.insert(handle)
                {
                    return Err(invalid("invalid Rust CPU backend binding"));
                }
                Ok(json!({"UseRefAvailable":false}))
            }
            "call" => self.call(action),
            "scalar" => {
                let handle = id(&action["Scalar"])?;
                let value = self
                    .scalars
                    .get(&handle)
                    .ok_or_else(|| invalid("Rust scalar missing"))?
                    .clone();
                if action["Release"] == true {
                    self.scalars.remove(&handle);
                }
                Ok(json!({"Value":value}))
            }
            "context-snapshot" => {
                let context = id(&action["Context"])?;
                let ids = self.backend.context_tensors(context);
                let mut aliases = tensor_aliases(&self.tensors);
                for &tensor in &ids {
                    let canonical = canonical(&aliases, tensor);
                    if let std::collections::btree_map::Entry::Vacant(entry) =
                        self.tensors.entry(canonical)
                    {
                        entry.insert(Some(tensor));
                        aliases.entry(tensor).or_default().push(canonical);
                    }
                }
                Ok(
                    json!({"Tensors":ids.into_iter().map(|tensor|metadata(&self.backend,&aliases,tensor)).collect::<Result<Vec<_>>>()?}),
                )
            }
            "tensor-snapshot" => {
                let tensor = self.tensor(&action["Tensor"])?;
                let mut aliases = tensor_aliases(&self.tensors);
                // Preserve complete snapshots' alias registration and source
                // identities while only materializing the requested metadata.
                for &context in &self.contexts {
                    for tensor in self.backend.context_tensors(context) {
                        let canonical = canonical(&aliases, tensor);
                        if let std::collections::btree_map::Entry::Vacant(entry) =
                            self.tensors.entry(canonical)
                        {
                            entry.insert(Some(tensor));
                            aliases.entry(tensor).or_default().push(canonical);
                        }
                    }
                }
                Ok(json!({"Tensor":metadata(&self.backend,&aliases,tensor)?}))
            }
            "graph-snapshot" => {
                let graph = lookup(&self.graphs, &action["Graph"])?;
                let (nodes, leaves) = self.backend.graph_nodes(graph)?;
                let gradients = nodes
                    .iter()
                    .map(|&tensor| {
                        Ok((
                            tensor,
                            self.backend.graph_gradient(graph, tensor)?,
                            self.backend.graph_gradient_accumulator(graph, tensor)?,
                        ))
                    })
                    .collect::<Result<Vec<_>>>()?;
                let mut aliases = tensor_aliases(&self.tensors);
                let exposed = nodes.iter().chain(leaves).copied().chain(
                    gradients
                        .iter()
                        .flat_map(|&(_, gradient, accumulator)| [gradient, accumulator])
                        .flatten(),
                );
                for tensor in exposed {
                    if aliases.contains_key(&tensor) {
                        continue;
                    }
                    let mut handle = canonical(&aliases, tensor);
                    while self.tensors.contains_key(&handle) {
                        handle = handle
                            .checked_add(1)
                            .ok_or_else(|| invalid("Rust snapshot tensor handles exhausted"))?;
                    }
                    self.tensors.insert(handle, Some(tensor));
                    aliases.insert(tensor, vec![handle]);
                }
                let gradients = nodes
                    .iter()
                    .map(|&tensor| {
                        let gradient = self.backend.graph_gradient(graph, tensor)?;
                        let accumulator = self.backend.graph_gradient_accumulator(graph, tensor)?;
                        Ok(json!({
                            "Tensor": canonical(&aliases, tensor),
                            "Gradient": gradient.map(|id| canonical(&aliases, id)),
                            "GradientAccumulator": accumulator.map(|id| canonical(&aliases, id)),
                        }))
                    })
                    .collect::<Result<Vec<_>>>()?;
                Ok(
                    json!({"Nodes":nodes.iter().map(|&id|canonical(&aliases,id)).collect::<Vec<_>>(),"Leaves":leaves.iter().map(|&id|canonical(&aliases,id)).collect::<Vec<_>>(),"Gradients":gradients}),
                )
            }
            "allocate" => {
                self.backend.allocate(id(&action["Context"])?)?;
                if !self.buffers.insert(id(&action["Result"]["Buffer"])?) {
                    return Err(invalid("duplicate Rust buffer"));
                }
                Ok(json!({}))
            }
            "supports" => {
                Ok(json!({"Supported":self.backend.supports(self.tensor(&action["Tensor"])?)?}))
            }
            "backend-features" => Ok(
                json!({"ProcedureAvailable":true,"Features":[{"Name":"SAFE_RUST","Value":"1"}],"RegistryName":"Rust CPU","BackendName":"Rust CPU","DeviceName":"CPU"}),
            ),
            "tensor-set" | "tensor-get" => {
                let tensor = self.tensor(&action["Tensor"])?;
                let offset = unsigned(&action["TensorOffset"])?;
                let bytes = unsigned(&action["Bytes"])?;
                let file_offset = unsigned(&action["FileOffset"])?;
                let path = self.file(&action["File"])?;
                if action["Op"] == "tensor-set" {
                    let data = self.read(&path, file_offset, bytes)?;
                    self.backend.set(tensor, offset, &data)?;
                    Ok(json!({}))
                } else {
                    self.write(
                        &path,
                        file_offset,
                        self.backend.get(tensor, offset, bytes)?,
                        action["Truncate"].as_bool().unwrap_or(file_offset == 0),
                    )?;
                    Ok(json!({"Bytes":bytes}))
                }
            }
            "tensor-write" => {
                let tensor = self.tensor(&action["Tensor"])?;
                let fields = action["Fields"]
                    .as_object()
                    .ok_or_else(|| invalid("Rust tensor fields must be an object"))?;
                for (field, value) in fields {
                    match field.as_str() {
                        "NB" => {
                            let values = array(value)?;
                            if values.len() != 4 {
                                return Err(invalid("Rust strides need four dimensions"));
                            }
                            for (axis, value) in values.iter().enumerate() {
                                self.backend.info_mut(tensor)?.strides[axis] = unsigned(value)?;
                            }
                        }
                        "Op" => {
                            self.backend.info_mut(tensor)?.operation =
                                i32::try_from(integer(value)?)?;
                        }
                        "Sources" => {
                            let slots = array(value)?
                                .iter()
                                .map(|value| {
                                    if value.is_null() {
                                        Ok(None)
                                    } else {
                                        Ok(Some(self.tensor(value)?))
                                    }
                                })
                                .collect::<Result<Vec<_>>>()?;
                            let info = self.backend.info_mut(tensor)?;
                            info.source_slots = slots
                                .try_into()
                                .map_err(|_| invalid("Rust backend sources require ten slots"))?;
                            info.sources = info.source_slots.iter().flatten().copied().collect();
                        }
                        "OpParamsHex" => {
                            self.backend.info_mut(tensor)?.parameters = unhex(value)?
                                .try_into()
                                .map_err(|_| invalid("Rust op params need 64 bytes"))?;
                        }
                        _ => return Err(invalid("Rust tensor mutable field unknown")),
                    }
                }
                Ok(json!({}))
            }
            "compare" => {
                let graph = lookup(&self.graphs, &action["Graph"])?;
                let prefix = self.file(&action["Prefix"])?;
                let maximum = unsigned(&action["MaxOutputBytes"])?;
                let aliases = tensor_aliases(&self.tensors);
                let mut pairs = Vec::new();
                let mut total = 0_usize;
                let selected = if action["Nodes"].is_null() {
                    None
                } else {
                    Some(
                        array(&action["Nodes"])?
                            .iter()
                            .map(|value| self.tensor(value))
                            .collect::<Result<Vec<_>>>()?,
                    )
                };
                self.backend.compare(graph,|index,tensor,left,right|{
                    if selected.as_ref().is_some_and(|selected|!selected.contains(&tensor)){return Ok(());}
                    let bytes=left.info(tensor)?.bytes()?;total=total.checked_add(2*bytes).ok_or_else(||invalid("Rust comparison extent overflow"))?;if total>maximum{return Err(invalid("Rust comparison output exceeds bound"));}
                    let left_path=PathBuf::from(format!("{}-{}-left.bin",prefix.display(),pairs.len()));let right_path=PathBuf::from(format!("{}-{}-right.bin",prefix.display(),pairs.len()));fs::write(&left_path,left.get(tensor,0,bytes)?)?;fs::write(&right_path,right.get(tensor,0,bytes)?)?;
                    pairs.push(json!({"Index":index,"Left":metadata(left,&aliases,tensor)?,"Right":metadata(right,&aliases,tensor)?,"LeftFile":left_path,"RightFile":right_path,"Bytes":bytes}));Ok(())
                })?;
                let mut captures = Vec::new();
                for handle in array(&action["CaptureTensors"])? {
                    let tensor = self.tensor(handle)?;
                    let bytes = self.backend.info(tensor)?.bytes()?;
                    total = total
                        .checked_add(bytes)
                        .ok_or_else(|| invalid("Rust comparison extent overflow"))?;
                    if total > maximum {
                        return Err(invalid("Rust comparison capture exceeds bound"));
                    }
                    let path =
                        PathBuf::from(format!("{}-capture-{}.bin", prefix.display(), id(handle)?));
                    fs::write(&path, self.backend.get(tensor, 0, bytes)?)?;
                    captures.push(json!({"Tensor":handle,"File":path,"Bytes":bytes}));
                }
                Ok(json!({"Compared":true,"Pairs":pairs,"Captures":captures}))
            }
            "type-info" => {
                let layout = type_layout(i32::try_from(integer(&action["Type"])?)?)?;
                Ok(
                    json!({"Name":layout.name,"BlockSize":layout.block,"TypeSize":layout.bytes,"Quantized":layout.block!=1,"RequiresImportance":false,"HasToFloat":true}),
                )
            }
            "quantize-file" => {
                let kind = i32::try_from(integer(&action["Type"])?)?;
                let layout = type_layout(kind)?;
                let path = self.file(&action["InputFile"])?;
                let start = unsigned(&action["Start"])?;
                let rows = unsigned(&action["Rows"])?;
                let width = unsigned(&action["ElementsPerRow"])?;
                let count = rows
                    .checked_mul(width)
                    .ok_or_else(|| invalid("Rust quantize extent overflow"))?;
                if rows == 0
                    || width == 0
                    || !start.is_multiple_of(width)
                    || !width.is_multiple_of(layout.block)
                {
                    return Err(invalid("Rust quantizer row/block extent differs"));
                }
                let data = self.read(&path, start * 4, count * 4)?;
                let input = self.backend.tensor(u32::MAX, kind, [width, rows, 1, 1])?;
                let values = data
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|bytes| f32::from_le_bytes(*bytes))
                    .collect::<Vec<_>>();
                let importance = if action["ImportanceFile"].is_null() {
                    None
                } else {
                    let path = self.file(&action["ImportanceFile"])?;
                    let data = self.read(&path, 0, width * 4)?;
                    Some(
                        data.as_chunks::<4>()
                            .0
                            .iter()
                            .map(|bytes| f32::from_le_bytes(*bytes))
                            .collect::<Vec<_>>(),
                    )
                };
                let mut data = Vec::new();
                for row in values.chunks_exact(width) {
                    let bytes = if let Some(importance) = &importance {
                        bonsai_prism_runtime::kernels::quantize_row_weighted(
                            u32::try_from(kind)?,
                            row,
                            importance,
                        )?
                    } else {
                        bonsai_prism_runtime::kernels::quantize_row(u32::try_from(kind)?, row)?
                    };
                    data.extend_from_slice(&bytes);
                }
                self.backend.set(input, 0, &data)?;
                let bytes = data.len();
                let output = self.file(&action["OutputFile"])?;
                self.write(
                    &output,
                    start / layout.block * layout.bytes,
                    self.backend.get(input, 0, bytes)?,
                    false,
                )?;
                Ok(json!({"Written":bytes}))
            }
            "dequantize-file" => {
                let kind = i32::try_from(integer(&action["Type"])?)?;
                let count = unsigned(&action["Count"])?;
                let input = self.backend.tensor(u32::MAX, kind, [count, 1, 1, 1])?;
                let bytes = self.backend.info(input)?.bytes()?;
                let path = self.file(&action["InputFile"])?;
                let data = self.read(&path, 0, bytes)?;
                self.backend.set(input, 0, &data)?;
                let values = self.backend.values(input)?;
                let data = values
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect::<Vec<_>>();
                let output = self.file(&action["OutputFile"])?;
                self.write(&output, 0, &data, true)?;
                Ok(json!({"Bytes":data.len()}))
            }
            "clock" => Ok(json!({"Microseconds":self.started.elapsed().as_micros()})),
            "hardware-concurrency" => {
                Ok(json!({"Threads":std::thread::available_parallelism()?.get()}))
            }
            other => Err(invalid(format!("Rust backend action unsupported: {other}"))),
        }
    }
}

fn actions(state: &State, value: &Value) -> Result<Vec<Value>> {
    let path = state.file(value)?;
    let metadata = fs::metadata(&path)?;
    if !metadata.is_file() || metadata.len() > MAX_ACTION_BYTES {
        return Err(invalid("Rust backend action file exceeds bound"));
    }
    let data = state.read(&path, 0, usize::try_from(metadata.len())?)?;
    let value: Value = serde_json::from_slice(&data)?;
    let values = array(&value)?;
    if values.len() > MAX_ACTIONS {
        return Err(invalid("Rust backend action count exceeds bound"));
    }
    Ok(values.to_vec())
}

pub(super) fn execute(operation: &Value) -> Result<Value> {
    let mut state = State::open(
        Path::new(text(&operation["Directory"])?),
        unsigned(&operation["MaxFileBytes"])?,
    )?;
    let mut supplied = if operation.get("ActionsFile").is_some() {
        actions(&state, &operation["ActionsFile"])?
    } else {
        array(&operation["Actions"])?.to_vec()
    };
    let mut total = 0_usize;
    let mut trace = Vec::new();
    loop {
        let mut continuation = None;
        for (index, action) in supplied.iter().enumerate() {
            total += 1;
            if total > MAX_ACTIONS {
                return Err(invalid("Rust backend cumulative actions exceed bound"));
            }
            if action["Op"] == "checkpoint" {
                if index + 1 != supplied.len() {
                    return Err(invalid("Rust backend checkpoint must end batch"));
                }
                let output = state.file(&action["OutputFile"])?;
                let next = state.file(&action["NextFile"])?;
                if output == next || output.try_exists()? || next.try_exists()? {
                    return Err(invalid("Rust backend checkpoint files must be fresh"));
                }
                trace.push(json!({}));
                let encoded = serde_json::to_vec(&json!({"Trace":trace}))?;
                if encoded.len() > 16 * 1024 * 1024 {
                    return Err(invalid("Rust backend checkpoint exceeds observation bound"));
                }
                let temporary = output.with_extension("pending");
                let mut file = OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(&temporary)?;
                file.write_all(&encoded)?;
                file.sync_all()?;
                drop(file);
                fs::rename(temporary, output)?;
                let milliseconds = unsigned(&action["TimeoutMilliseconds"])?;
                if !(1..=3_600_000).contains(&milliseconds) {
                    return Err(invalid("Rust backend continuation timeout outside bound"));
                }
                let deadline = Instant::now() + Duration::from_millis(milliseconds as u64);
                while !next.try_exists()? {
                    if Instant::now() >= deadline {
                        return Err(invalid("Rust backend continuation timeout"));
                    }
                    std::thread::sleep(Duration::from_millis(1));
                }
                continuation = Some(actions(&state, &json!(next))?);
                trace = Vec::new();
                break;
            }
            trace.push(
                state.run(action).map_err(|error| {
                    invalid(format!("Rust backend action {}: {error}", total - 1))
                })?,
            );
        }
        if let Some(next) = continuation {
            supplied = next;
        } else {
            return Ok(json!({"Trace":trace}));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_alias_index_preserves_handle_order_nulls_and_unbound_identity() {
        let tensors = BTreeMap::from([
            (2, Some(TensorId(1))),
            (3, Some(TensorId(0))),
            (7, None),
            (9, Some(TensorId(0))),
            (0x8000_0000, Some(TensorId(0))),
        ]);
        let aliases = tensor_aliases(&tensors);
        assert_eq!(aliases[&TensorId(0)], [3, 9, 0x8000_0000]);
        assert_eq!(aliases[&TensorId(1)], [2]);
        assert_eq!(aliases.len(), 2);
        assert_eq!(canonical(&aliases, TensorId(0)), 3);
        assert_eq!(canonical(&aliases, TensorId(1)), 2);
        assert_eq!(canonical(&aliases, TensorId(17)), 0x8000_0011);
    }

    #[test]
    fn graph_snapshot_exports_owned_gradients_with_registered_aliases() -> Result<()> {
        let mut state = State::open(&std::env::temp_dir(), 1024)?;
        let parameter = state.backend.tensor(1, 0, [2, 1, 1, 1])?;
        state
            .backend
            .call("ggml_set_param", &[Argument::Tensor(parameter)])?;
        let Observation::Tensor(Some(square)) = state.backend.call(
            "ggml_sqr",
            &[Argument::Context(1), Argument::Tensor(parameter)],
        )?
        else {
            return Err(invalid("square tensor missing"));
        };
        let Observation::Tensor(Some(loss)) = state.backend.call(
            "ggml_sum",
            &[Argument::Context(1), Argument::Tensor(square)],
        )?
        else {
            return Err(invalid("loss tensor missing"));
        };
        state
            .backend
            .call("ggml_set_loss", &[Argument::Tensor(loss)])?;
        let Observation::Graph(graph) = state.backend.call(
            "ggml_new_graph_custom",
            &[
                Argument::Context(1),
                Argument::Integer(64),
                Argument::Bool(true),
            ],
        )?
        else {
            return Err(invalid("gradient graph missing"));
        };
        state.backend.call(
            "ggml_build_forward_expand",
            &[Argument::Graph(graph), Argument::Tensor(loss)],
        )?;
        state.graphs.insert(4, graph);
        state.tensors.insert(10, Some(parameter));
        state.tensors.insert(11, Some(loss));
        let before = state.run(&json!({"Op":"graph-snapshot","Graph":4}))?;
        assert!(array(&before["Gradients"])?.iter().all(|entry| {
            entry["Gradient"].is_null() && entry["GradientAccumulator"].is_null()
        }));
        state.backend.call(
            "ggml_build_backward_expand",
            &[Argument::Context(1), Argument::Graph(graph), Argument::Null],
        )?;
        let gradient = state
            .backend
            .graph_gradient(graph, parameter)?
            .ok_or_else(|| invalid("parameter gradient missing"))?;
        let occupied = canonical(&TensorAliases::new(), gradient);
        state.tensors.insert(occupied, Some(loss));
        let after = state.run(&json!({"Op":"graph-snapshot","Graph":4}))?;
        let entries = array(&after["Gradients"])?;
        let parameter = entries
            .iter()
            .find(|entry| entry["Tensor"] == 10)
            .ok_or_else(|| invalid("parameter snapshot missing"))?;
        assert_eq!(state.tensor(&parameter["Gradient"])?, gradient);
        assert_eq!(state.tensor(&json!(occupied))?, loss);
        state.tensors.insert(20, Some(gradient));
        let aliased = state.run(&json!({"Op":"graph-snapshot","Graph":4}))?;
        let aliased_parameter = array(&aliased["Gradients"])?
            .iter()
            .find(|entry| entry["Tensor"] == 10)
            .ok_or_else(|| invalid("aliased parameter snapshot missing"))?;
        assert_eq!(aliased_parameter["Gradient"], 20);
        let loss = entries
            .iter()
            .find(|entry| entry["Tensor"] == 11)
            .ok_or_else(|| invalid("loss snapshot missing"))?;
        assert!(loss["Gradient"].is_u64());
        assert_eq!(loss["Gradient"], loss["GradientAccumulator"]);
        Ok(())
    }

    #[test]
    fn direct_rust_sessions_enforce_checkpoint_slots_deadlines_and_poisoning() -> Result<()> {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos();
        let base = std::env::temp_dir();
        if !base.is_absolute() {
            return Err(invalid("Rust session temp directory must be absolute"));
        }
        let root = base.join(format!(
            "prism-rust-session-bounds-{}-{stamp}",
            std::process::id()
        ));
        fs::create_dir(&root)?;
        for (name, deadline, total, actions, success) in [
            (
                "last-slot",
                Instant::now() + Duration::from_secs(60),
                MAX_ACTIONS - 2,
                vec![json!({"Op":"clock"})],
                true,
            ),
            (
                "overflow",
                Instant::now() + Duration::from_secs(60),
                MAX_ACTIONS - 1,
                vec![json!({"Op":"clock"})],
                false,
            ),
            (
                "empty",
                Instant::now() + Duration::from_secs(60),
                1,
                vec![],
                false,
            ),
            (
                "expired",
                Instant::now(),
                1,
                vec![json!({"Op":"clock"})],
                false,
            ),
        ] {
            let directory = root.join(name);
            fs::create_dir(&directory)?;
            let mut session = GraphSession::open(&directory, 1024, deadline)?;
            session.total_actions = total;
            assert_eq!(session.exchange(&actions).is_ok(), success);
            if success {
                assert_eq!(session.total_actions, MAX_ACTIONS);
                let response: Value =
                    serde_json::from_slice(&fs::read(directory.join("observations.jsonl"))?)?;
                assert_eq!(array(&response["Trace"])?.len(), 1);
                assert!(session.exchange(&[json!({"Op":"clock"})]).is_err());
            }
            assert!(session.poisoned);
            assert!(session.exchange(&[json!({"Op":"clock"})]).is_err());
        }
        fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn direct_rust_trace_framing_enforces_the_exact_observation_cap() -> Result<()> {
        let overhead = encode_trace(&[json!("")])?.len();
        let mut values = [Value::String("x".repeat(MAX_OBSERVATION_BYTES - overhead))];
        assert_eq!(encode_trace(&values)?.len(), MAX_OBSERVATION_BYTES);
        let Value::String(value) = &mut values[0] else {
            return Err(invalid("trace fixture is not text"));
        };
        value.push('x');
        assert!(encode_trace(&values).is_err());
        Ok(())
    }
}
