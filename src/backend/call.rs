//! Typed graph construction and shape validation for production operations.

use super::{
    Argument, CpuBackend, Graph, GraphId, Observation, TensorId, TensorInfo, product, type_layout,
};
use crate::{Result, invalid};

fn at(args: &[Argument], index: usize) -> Result<&Argument> {
    args.get(index)
        .ok_or_else(|| invalid("CPU operation argument is missing"))
}
fn int(args: &[Argument], index: usize) -> Result<i64> {
    if let Argument::Integer(value) = at(args, index)? {
        Ok(*value)
    } else {
        Err(invalid("CPU operation requires an integer"))
    }
}
fn usize_arg(args: &[Argument], index: usize) -> Result<usize> {
    Ok(usize::try_from(int(args, index)?)?)
}
fn float(args: &[Argument], index: usize) -> Result<f32> {
    if let Argument::Float(value) = at(args, index)? {
        Ok(*value)
    } else {
        Err(invalid("CPU operation requires binary32"))
    }
}
fn boolean(args: &[Argument], index: usize) -> Result<bool> {
    if let Argument::Bool(value) = at(args, index)? {
        Ok(*value)
    } else {
        Err(invalid("CPU operation requires a boolean"))
    }
}
fn tensor(args: &[Argument], index: usize) -> Result<TensorId> {
    if let Argument::Tensor(value) = at(args, index)? {
        Ok(*value)
    } else {
        Err(invalid("CPU operation requires a tensor"))
    }
}
fn graph(args: &[Argument], index: usize) -> Result<GraphId> {
    if let Argument::Graph(value) = at(args, index)? {
        Ok(*value)
    } else {
        Err(invalid("CPU operation requires a graph"))
    }
}
fn string(args: &[Argument], index: usize) -> Result<Option<String>> {
    if let Argument::String(value) = at(args, index)? {
        Ok(value.clone())
    } else {
        Err(invalid("CPU operation requires a string"))
    }
}
fn context(args: &[Argument], index: usize) -> Result<u32> {
    if let Argument::Context(value) = at(args, index)? {
        Ok(*value)
    } else {
        Err(invalid("CPU operation requires a context"))
    }
}
fn put_int(params: &mut [u8; 64], index: usize, value: i32) {
    params[index * 4..index * 4 + 4].copy_from_slice(&value.to_le_bytes());
}
fn put_float(params: &mut [u8; 64], index: usize, value: f32) {
    params[index * 4..index * 4 + 4].copy_from_slice(&value.to_le_bytes());
}

fn convolution_extent(
    input: usize,
    kernel: usize,
    stride: i64,
    padding: i64,
    dilation: i64,
) -> Result<usize> {
    if stride <= 0 || dilation <= 0 || padding < 0 {
        return Err(invalid(
            "CPU convolution stride, padding or dilation differs",
        ));
    }
    let extent =
        (input as i128 + 2 * i128::from(padding) - i128::from(dilation) * (kernel as i128 - 1) - 1)
            / i128::from(stride)
            + 1;
    if extent <= 0 {
        return Err(invalid("CPU convolution kernel exceeds padded input"));
    }
    Ok(usize::try_from(extent)?)
}

/// Resolve the pinned numerical operation name.
#[must_use]
#[allow(
    clippy::too_many_lines,
    reason = "All pinned operation discriminants are kept in one auditable mapping."
)]
pub const fn operation_name(operation: i32) -> &'static str {
    match operation {
        0 => "NONE",
        1 => "DUP",
        2 => "ADD",
        3 => "ADD_ID",
        4 => "ADD1",
        5 => "ACC",
        6 => "SUB",
        7 => "MUL",
        8 => "DIV",
        9 => "SQR",
        10 => "SQRT",
        11 => "LOG",
        12 => "SIN",
        13 => "COS",
        14 => "SUM",
        15 => "SUM_ROWS",
        16 => "CUMSUM",
        17 => "MEAN",
        18 => "ARGMAX",
        19 => "COUNT_EQUAL",
        20 => "REPEAT",
        21 => "REPEAT_BACK",
        22 => "CONCAT",
        23 => "SILU_BACK",
        24 => "NORM",
        25 => "RMS_NORM",
        26 => "RMS_NORM_BACK",
        27 => "GROUP_NORM",
        28 => "L2_NORM",
        29 => "MUL_MAT",
        30 => "MUL_MAT_ID",
        31 => "OUT_PROD",
        32 => "SCALE",
        33 => "SET",
        34 => "CPY",
        35 => "CONT",
        36 => "RESHAPE",
        37 => "VIEW",
        38 => "PERMUTE",
        39 => "TRANSPOSE",
        40 => "GET_ROWS",
        41 => "GET_ROWS_BACK",
        42 => "SET_ROWS",
        43 => "DIAG",
        44 => "DIAG_MASK_INF",
        45 => "DIAG_MASK_ZERO",
        46 => "SOFT_MAX",
        47 => "SOFT_MAX_BACK",
        48 => "ROPE",
        49 => "ROPE_BACK",
        50 => "CLAMP",
        51 => "CONV_TRANSPOSE_1D",
        52 => "IM2COL",
        53 => "IM2COL_BACK",
        54 => "IM2COL_3D",
        55 => "COL2IM_1D",
        56 => "CONV_2D",
        57 => "CONV_3D",
        58 => "CONV_2D_DW",
        59 => "CONV_TRANSPOSE_2D",
        60 => "POOL_1D",
        61 => "POOL_2D",
        62 => "POOL_2D_BACK",
        63 => "UPSCALE",
        64 => "PAD",
        65 => "PAD_REFLECT_1D",
        66 => "ROLL",
        67 => "ARANGE",
        68 => "TIMESTEP_EMBEDDING",
        69 => "ARGSORT",
        70 => "TOP_K",
        71 => "LEAKY_RELU",
        72 => "TRI",
        73 => "FILL",
        74 => "FLASH_ATTN_EXT",
        75 => "FLASH_ATTN_BACK",
        76 => "SSM_CONV",
        77 => "SSM_SCAN",
        78 => "WIN_PART",
        79 => "WIN_UNPART",
        80 => "GET_REL_POS",
        81 => "ADD_REL_POS",
        82 => "RWKV_WKV6",
        83 => "GATED_LINEAR_ATTN",
        84 => "RWKV_WKV7",
        85 => "SOLVE_TRI",
        86 => "GATED_DELTA_NET",
        87 => "LIGHTNING_INDEXER",
        88 => "DSV4_HC_COMB",
        89 => "DSV4_HC_PRE",
        90 => "DSV4_HC_POST",
        91 => "UNARY",
        92 => "MAP_CUSTOM1",
        93 => "MAP_CUSTOM2",
        94 => "MAP_CUSTOM3",
        95 => "CUSTOM",
        96 => "CROSS_ENTROPY_LOSS",
        97 => "CROSS_ENTROPY_LOSS_BACK",
        98 => "OPT_STEP_ADAMW",
        99 => "OPT_STEP_SGD",
        100 => "GLU",
        101 => "GATED_DELTA_NET_REPLAY",
        _ => "UNKNOWN",
    }
}

/// Resolve a pinned unary operation name.
#[must_use]
pub const fn unary_name(operation: i32) -> &'static str {
    match operation {
        0 => "ABS",
        1 => "SGN",
        2 => "NEG",
        3 => "STEP",
        4 => "TANH",
        5 => "ELU",
        6 => "RELU",
        7 => "SIGMOID",
        8 => "GELU",
        9 => "GELU_QUICK",
        10 => "SILU",
        11 => "HARDSWISH",
        12 => "HARDSIGMOID",
        13 => "EXP",
        14 => "EXPM1",
        15 => "SOFTPLUS",
        16 => "GELU_ERF",
        17 => "XIELU",
        18 => "FLOOR",
        19 => "CEIL",
        20 => "ROUND",
        21 => "TRUNC",
        _ => "UNKNOWN",
    }
}

impl CpuBackend {
    #[allow(
        clippy::too_many_arguments,
        reason = "Tensor construction keeps the complete pinned metadata contract explicit."
    )]
    pub(super) fn derived(
        &mut self,
        ctx: u32,
        kind: i32,
        shape: [usize; 4],
        operation: i32,
        sources: Vec<TensorId>,
        params: [u8; 64],
        view: Option<(TensorId, [usize; 4], usize)>,
    ) -> Result<Observation> {
        let id = self.tensor(ctx, kind, shape)?;
        let view = if let Some((source, strides, offset)) = view {
            let (root, relative) = self.view_root(source)?;
            Some((
                root,
                strides,
                relative
                    .checked_add(offset)
                    .ok_or_else(|| invalid("CPU view offset overflow"))?,
            ))
        } else {
            None
        };
        let info = self.info_mut(id)?;
        info.operation = operation;
        for (slot, &source) in info.source_slots.iter_mut().zip(&sources) {
            *slot = Some(source);
        }
        info.sources = sources;
        info.parameters = params;
        if let Some((source, strides, offset)) = view {
            info.view = Some(source);
            info.strides = strides;
            info.offset = offset;
        }
        Ok(Observation::Tensor(Some(id)))
    }

    /// Invoke a typed production graph operation from the pinned GGML contract.
    ///
    /// Graph construction and execution use the same safe storage and kernels as
    /// direct runtime calls. No test programs or reference answers are consulted.
    ///
    /// # Errors
    /// Rejects unsupported calls, wrong argument types, invalid shapes, storage
    /// overflow, unknown identities, and numerical execution failures.
    #[allow(
        clippy::too_many_lines,
        reason = "The allowlist keeps operation shape rules adjacent to their typed dispatch."
    )]
    pub fn call(&mut self, name: &str, args: &[Argument]) -> Result<Observation> {
        match name {
            "ggml_type_size" | "ggml_blck_size" | "ggml_type_name" | "ggml_is_quantized" => {
                let layout = type_layout(i32::try_from(int(args, 0)?)?)?;
                return Ok(match name {
                    "ggml_type_size" => Observation::Integer(layout.bytes as i64),
                    "ggml_blck_size" => Observation::Integer(layout.block as i64),
                    "ggml_is_quantized" => Observation::Bool(layout.block != 1),
                    _ => Observation::String(Some(layout.name.into())),
                });
            }
            "ggml_tensor_overhead" | "ggml_graph_overhead" | "ggml_graph_overhead_custom" => {
                return Ok(Observation::Integer(size_of::<TensorInfo>() as i64));
            }
            "ggml_op_name" => {
                return Ok(Observation::String(Some(
                    operation_name(i32::try_from(int(args, 0)?)?).into(),
                )));
            }
            "ggml_unary_op_name" => {
                return Ok(Observation::String(Some(
                    unary_name(i32::try_from(int(args, 0)?)?).into(),
                )));
            }
            "ggml_nbytes" | "ggml_nelements" | "ggml_nrows" | "ggml_element_size"
            | "ggml_is_vector" | "ggml_is_matrix" | "ggml_get_name" | "ggml_op_desc" => {
                let info = self.info(tensor(args, 0)?)?;
                return Ok(match name {
                    "ggml_nbytes" => Observation::Integer(info.bytes()? as i64),
                    "ggml_nelements" => Observation::Integer(info.elements() as i64),
                    "ggml_nrows" => Observation::Integer(info.rows() as i64),
                    "ggml_element_size" => {
                        Observation::Integer(type_layout(info.kind)?.bytes as i64)
                    }
                    "ggml_is_vector" => {
                        Observation::Bool(info.shape[1..].iter().all(|&dim| dim == 1))
                    }
                    "ggml_is_matrix" => {
                        Observation::Bool(info.shape[2..].iter().all(|&dim| dim == 1))
                    }
                    "ggml_get_name" => Observation::String(Some(info.name.clone())),
                    _ => Observation::String(Some(
                        if info.operation == 91 {
                            unary_name(i32::from_le_bytes(info.parameters[..4].try_into()?))
                        } else {
                            operation_name(info.operation)
                        }
                        .into(),
                    )),
                });
            }
            "ggml_are_same_shape" => {
                return Ok(Observation::Bool(
                    self.info(tensor(args, 0)?)?.shape == self.info(tensor(args, 1)?)?.shape,
                ));
            }
            "ggml_get_tensor" => {
                let ctx = context(args, 0)?;
                let name =
                    string(args, 1)?.ok_or_else(|| invalid("CPU tensor lookup name is null"))?;
                return Ok(Observation::Tensor(
                    self.context_tensors(ctx)
                        .into_iter()
                        .find(|&id| self.info(id).is_ok_and(|info| info.name == name)),
                ));
            }
            "ggml_set_name" => {
                let id = tensor(args, 0)?;
                self.info_mut(id)?.name =
                    string(args, 1)?.ok_or_else(|| invalid("CPU tensor name is null"))?;
                return Ok(Observation::Tensor(Some(id)));
            }
            "ggml_set_param" | "ggml_set_loss" => {
                let info = self.info_mut(tensor(args, 0)?)?;
                if name == "ggml_set_param" {
                    if info.operation != 0 {
                        return Err(invalid("CPU parameter must be a leaf tensor"));
                    }
                    info.flags |= 16;
                } else {
                    if info.kind != 0 || info.elements() != 1 {
                        return Err(invalid("CPU loss must be scalar F32"));
                    }
                    info.flags |= 8;
                }
                return Ok(Observation::Void);
            }
            "ggml_new_graph" | "ggml_new_graph_custom" => {
                let capacity = if name == "ggml_new_graph" {
                    2048
                } else {
                    usize_arg(args, 1)?
                };
                if capacity == 0 || capacity > 32768 {
                    return Err(invalid("CPU graph capacity outside bounds"));
                }
                let id = GraphId(self.graphs.len());
                self.graphs.push(Graph {
                    nodes: Vec::new(),
                    leaves: Vec::new(),
                    capacity,
                    gradients_enabled: name == "ggml_new_graph_custom" && boolean(args, 2)?,
                    ..Graph::default()
                });
                return Ok(Observation::Graph(id));
            }
            "ggml_build_forward_expand" => {
                self.expand(graph(args, 0)?, tensor(args, 1)?)?;
                return Ok(Observation::Void);
            }
            "ggml_build_backward_expand" => {
                if !matches!(at(args, 2)?, Argument::Null) {
                    return Err(invalid(
                        "CPU typed backward dispatch requires null accumulators; use build_backward for owned accumulators",
                    ));
                }
                self.build_backward(context(args, 0)?, graph(args, 1)?, None)?;
                return Ok(Observation::Void);
            }
            "ggml_graph_get_grad" | "ggml_graph_get_grad_acc" => {
                let id = graph(args, 0)?;
                let tensor = tensor(args, 1)?;
                return Ok(Observation::Tensor(if name == "ggml_graph_get_grad" {
                    self.graph_gradient(id, tensor)?
                } else {
                    self.graph_gradient_accumulator(id, tensor)?
                }));
            }
            "ggml_graph_add_node" => {
                let id = tensor(args, 1)?;
                self.info(id)?;
                let graph = &mut self.graphs[graph(args, 0)?.0];
                if graph.nodes.len() >= graph.capacity {
                    return Err(invalid("CPU graph capacity exceeded"));
                }
                graph.nodes.push(id);
                return Ok(Observation::Void);
            }
            "ggml_graph_n_nodes" | "ggml_graph_size" => {
                let graph = self
                    .graphs
                    .get(graph(args, 0)?.0)
                    .ok_or_else(|| invalid("unknown CPU graph"))?;
                return Ok(Observation::Integer(if name == "ggml_graph_size" {
                    graph.capacity
                } else {
                    graph.nodes.len()
                } as i64));
            }
            "ggml_graph_cpy" => {
                let source = graph(args, 0)?;
                let destination = graph(args, 1)?;
                let (nodes, leaves) = self.graph_nodes(source)?;
                let nodes = nodes.to_vec();
                let leaves = leaves.to_vec();
                let gradients = self.graphs[source.0].gradients.clone();
                let accumulators = self.graphs[source.0].gradient_accumulators.clone();
                let gradients_enabled = self.graphs[source.0].gradients_enabled;
                let graph = self
                    .graphs
                    .get_mut(destination.0)
                    .ok_or_else(|| invalid("unknown CPU destination graph"))?;
                if nodes.len() > graph.capacity {
                    return Err(invalid("CPU graph copy exceeds capacity"));
                }
                if gradients_enabled && !graph.gradients_enabled {
                    return Err(invalid("CPU graph copy requires gradient storage"));
                }
                graph.nodes = nodes;
                graph.leaves = leaves;
                graph.gradients = gradients;
                graph.gradient_accumulators = accumulators;
                return Ok(Observation::Void);
            }
            "ggml_graph_reset" => {
                self.reset_graph(graph(args, 0)?)?;
                return Ok(Observation::Void);
            }
            "ggml_rope_set_offset" => {
                let id = tensor(args, 0)?;
                let offset = i32::try_from(int(args, 1)?)?;
                if offset < 0 || offset % 2 != 0 {
                    return Err(invalid("CPU rotary offset differs"));
                }
                put_int(&mut self.info_mut(id)?.parameters, 15, offset);
                return Ok(Observation::Tensor(Some(id)));
            }
            "ggml_flash_attn_ext_set_prec" => {
                let id = tensor(args, 0)?;
                let precision = i32::try_from(int(args, 1)?)?;
                let info = self.info_mut(id)?;
                if info.operation != 74 {
                    return Err(invalid(
                        "CPU attention precision requires an attention operation",
                    ));
                }
                put_int(&mut info.parameters, 3, precision);
                return Ok(Observation::Void);
            }
            "ggml_gated_delta_net_set_raw_gates" => {
                let id = tensor(args, 0)?;
                let bias = tensor(args, 1)?;
                let weight = tensor(args, 2)?;
                let info = self.info(id)?;
                if info.operation != 86 || self.info(info.sources[3])?.shape[0] != 1 {
                    return Err(invalid(
                        "CPU raw gates require scalar-gated delta attention",
                    ));
                }
                let heads = self.info(info.sources[2])?.shape[1];
                for input in [bias, weight] {
                    let info = self.info(input)?;
                    if info.kind != 0 || info.elements() != heads || !info.contiguous() {
                        return Err(invalid("CPU raw gate weight extent differs"));
                    }
                }
                let info = self.info_mut(id)?;
                put_int(&mut info.parameters, 1, 1);
                info.source_slots[7] = Some(bias);
                info.source_slots[8] = Some(weight);
                info.sources = info.source_slots.iter().flatten().copied().collect();
                return Ok(Observation::Void);
            }
            "ggml_flash_attn_ext_add_sinks" | "ggml_soft_max_add_sinks" => {
                let id = tensor(args, 0)?;
                let flash = name == "ggml_flash_attn_ext_add_sinks";
                let slot = if flash { 4 } else { 2 };
                let sink = match at(args, 1)? {
                    Argument::Null => None,
                    Argument::Tensor(id) => Some(*id),
                    _ => return Err(invalid("CPU attention sink argument differs")),
                };
                if let Some(sink) = sink {
                    let si = self.info(sink)?;
                    let info = self.info(id)?;
                    let heads = if flash {
                        self.info(info.sources[0])?.shape[2]
                    } else {
                        info.shape[2]
                    };
                    if si.kind != 0 || si.shape[0] != heads {
                        return Err(invalid("CPU attention sink shape differs"));
                    }
                }
                let info = self.info_mut(id)?;
                if info.operation != if flash { 74 } else { 46 } {
                    return Err(invalid("CPU attention sink operation differs"));
                }
                info.source_slots[slot] = sink;
                info.sources = info.source_slots.iter().flatten().copied().collect();
                return Ok(Observation::Void);
            }
            "ggml_backend_graph_compute" => {
                self.compute(graph(args, 1)?)?;
                return Ok(Observation::Integer(0));
            }
            "ggml_mul_mat_set_hint" | "ggml_backend_buffer_set_usage" => {
                return Ok(Observation::Void);
            }
            "ggml_new_tensor" | "ggml_new_tensor_1d" | "ggml_new_tensor_2d"
            | "ggml_new_tensor_3d" | "ggml_new_tensor_4d" => {
                let ctx = context(args, 0)?;
                let kind = i32::try_from(int(args, 1)?)?;
                let mut shape = [1; 4];
                if name == "ggml_new_tensor" {
                    let dims = usize_arg(args, 2)?;
                    let Argument::Integers(values) = at(args, 3)? else {
                        return Err(invalid("CPU shape requires an integer list"));
                    };
                    if !(1..=4).contains(&dims) || values.len() < dims {
                        return Err(invalid("CPU shape rank outside bounds"));
                    }
                    for axis in 0..dims {
                        shape[axis] = usize::try_from(values[axis])?;
                    }
                } else {
                    for (axis, dim) in shape.iter_mut().enumerate().take(args.len() - 2) {
                        *dim = usize_arg(args, axis + 2)?;
                    }
                }
                return Ok(Observation::Tensor(Some(self.tensor(ctx, kind, shape)?)));
            }
            "ggml_arange" => {
                let ctx = context(args, 0)?;
                let start = float(args, 1)?;
                let end = float(args, 2)?;
                let step = float(args, 3)?;
                if !start.is_finite()
                    || !end.is_finite()
                    || !step.is_finite()
                    || step <= 0.0
                    || end <= start
                {
                    return Err(invalid("CPU arange bounds differ"));
                }
                let shape = [((end - start) / step).ceil() as usize, 1, 1, 1];
                let mut p = [0; 64];
                put_float(&mut p, 0, start);
                put_float(&mut p, 1, end);
                put_float(&mut p, 2, step);
                return self.derived(ctx, 0, shape, 67, Vec::new(), p, None);
            }
            _ => {}
        }
        let ctx = context(args, 0)?;
        let a = tensor(args, 1)?;
        let a_info = self.info(a)?.clone();
        let mut shape = a_info.shape;
        let mut kind = a_info.kind;
        let mut sources = vec![a];
        let mut p = [0; 64];
        let mut view = None;
        let mut inplace = name.ends_with("_inplace");
        let op = match name {
            "ggml_dup" => 1,
            "ggml_add" | "ggml_add_inplace" => 2,
            "ggml_add1" | "ggml_add1_inplace" => {
                let b = tensor(args, 2)?;
                let scalar = self.info(b)?;
                let packed = super::ops::packed_add1(kind) && scalar.kind == 0;
                if scalar.elements() != 1
                    || !(packed
                        || matches!((kind, scalar.kind), (0, 0) | (1, 0 | 1) | (30, 0 | 30)))
                    || a_info.strides[0] != type_layout(kind)?.bytes
                    || a_info.strides[1].checked_mul(shape[1]) != Some(a_info.strides[2])
                    || a_info.strides[2].checked_mul(shape[2]) != Some(a_info.strides[3])
                    // Pinned packed ADD1 uses i3*nb0 for the destination's
                    // fourth axis. Restrict that unqualified native domain.
                    || packed && shape[3] != 1
                {
                    return Err(invalid(
                        "CPU scalar add requires compatible padded storage and a supported scalar",
                    ));
                }
                sources.push(b);
                4
            }
            "ggml_win_part" => {
                let window = usize_arg(args, 2)?;
                if window == 0 || kind != 0 || shape[3] != 1 || !a_info.contiguous() {
                    return Err(invalid("CPU window partition geometry differs"));
                }
                let columns = shape[1].div_ceil(window);
                let rows = shape[2].div_ceil(window);
                shape = [
                    shape[0],
                    window,
                    window,
                    columns
                        .checked_mul(rows)
                        .ok_or_else(|| invalid("CPU window partition count overflow"))?,
                ];
                put_int(&mut p, 0, i32::try_from(columns)?);
                put_int(&mut p, 1, i32::try_from(rows)?);
                put_int(&mut p, 2, i32::try_from(window)?);
                78
            }
            "ggml_win_unpart" => {
                let width = usize_arg(args, 2)?;
                let height = usize_arg(args, 3)?;
                let window = usize_arg(args, 4)?;
                if window == 0
                    || width == 0
                    || height == 0
                    || kind != 0
                    || shape[1] != window
                    || shape[2] != window
                    || width.div_ceil(window).checked_mul(height.div_ceil(window)) != Some(shape[3])
                    || !a_info.contiguous()
                {
                    return Err(invalid("CPU window unpartition geometry differs"));
                }
                shape = [shape[0], width, height, 1];
                put_int(&mut p, 0, i32::try_from(window)?);
                79
            }
            "ggml_get_rel_pos" => {
                let query = usize_arg(args, 2)?;
                let key = usize_arg(args, 3)?;
                if query == 0
                    || query != key
                    || query.checked_mul(2).and_then(|n| n.checked_sub(1)) != Some(shape[1])
                    || !matches!(kind, 1 | 30)
                    || !a_info.contiguous()
                {
                    return Err(invalid("CPU relative position geometry differs"));
                }
                shape = [shape[0], key, query, 1];
                kind = 1;
                80
            }
            "ggml_add_rel_pos" | "ggml_add_rel_pos_inplace" => {
                let width = tensor(args, 2)?;
                let height = tensor(args, 3)?;
                let wi = self.info(width)?;
                let hi = self.info(height)?;
                if kind != 0
                    || wi.kind != 0
                    || hi.kind != 0
                    || wi.shape != hi.shape
                    || !a_info.contiguous()
                    || !wi.contiguous()
                    || !hi.contiguous()
                    || wi.shape[3] != shape[2]
                    || wi.shape[0].checked_mul(wi.shape[0]) != Some(shape[0])
                    || wi.shape[1].checked_mul(wi.shape[2]) != Some(shape[1])
                {
                    return Err(invalid("CPU relative position addition geometry differs"));
                }
                sources.extend([width, height]);
                put_int(&mut p, 0, i32::from(inplace));
                81
            }
            "ggml_add_id" => {
                let b = tensor(args, 2)?;
                let ids = tensor(args, 3)?;
                let bi = self.info(b)?;
                let indices = self.info(ids)?;
                if shape[0] != bi.shape[0]
                    || shape[1] != indices.shape[0]
                    || shape[2] != indices.shape[1]
                    || indices.kind != 26
                {
                    return Err(invalid("CPU indexed add geometry differs"));
                }
                sources.push(b);
                sources.push(ids);
                3
            }
            "ggml_sub" | "ggml_sub_inplace" => 6,
            "ggml_mul" | "ggml_mul_inplace" => 7,
            "ggml_div" | "ggml_div_inplace" => 8,
            "ggml_sqr" => 9,
            "ggml_sqrt" => 10,
            "ggml_log" => 11,
            "ggml_sin" => 12,
            "ggml_cos" => 13,
            "ggml_sum" => {
                shape = [1; 4];
                14
            }
            "ggml_sum_rows" => {
                shape[0] = 1;
                15
            }
            "ggml_cumsum" => 16,
            "ggml_mean" => {
                shape[0] = 1;
                17
            }
            "ggml_argmax" => {
                shape = [a_info.shape[1], 1, 1, 1];
                kind = 26;
                18
            }
            "ggml_count_equal" => {
                sources.push(tensor(args, 2)?);
                shape = [1; 4];
                kind = 27;
                19
            }
            "ggml_repeat" | "ggml_repeat_back" => {
                let b = tensor(args, 2)?;
                shape = self.info(b)?.shape;
                if name == "ggml_repeat" { 20 } else { 21 }
            }
            "ggml_repeat_4d" => {
                for (axis, dim) in shape.iter_mut().enumerate() {
                    *dim = usize_arg(args, axis + 2)?;
                }
                20
            }
            "ggml_concat" => {
                let b = tensor(args, 2)?;
                let bs = self.info(b)?.shape;
                let axis = usize_arg(args, 3)?;
                if axis >= 4 || (0..4).any(|i| i != axis && shape[i] != bs[i]) {
                    return Err(invalid("CPU concat shapes differ"));
                }
                shape[axis] = shape[axis]
                    .checked_add(bs[axis])
                    .ok_or_else(|| invalid("CPU concat extent overflow"))?;
                sources.push(b);
                put_int(&mut p, 0, axis as i32);
                22
            }
            "ggml_silu_back" => {
                sources.push(tensor(args, 2)?);
                23
            }
            "ggml_norm" => {
                put_float(&mut p, 0, float(args, 2)?);
                24
            }
            "ggml_rms_norm" | "ggml_rms_norm_inplace" => {
                put_float(&mut p, 0, float(args, 2)?);
                25
            }
            "ggml_rms_norm_back" => {
                sources.push(tensor(args, 2)?);
                put_float(&mut p, 0, float(args, 3)?);
                26
            }
            "ggml_group_norm" => {
                let groups = int(args, 2)?;
                if groups <= 0 || groups > shape[2] as i64 {
                    return Err(invalid("CPU group count outside bounds"));
                }
                put_int(&mut p, 0, i32::try_from(groups)?);
                put_float(&mut p, 1, float(args, 3)?);
                27
            }
            "ggml_l2_norm" => {
                put_float(&mut p, 0, float(args, 2)?);
                28
            }
            "ggml_mul_mat" => {
                let b = tensor(args, 2)?;
                let bs = self.info(b)?.shape;
                if shape[0] != bs[0]
                    || shape[2] == 0
                    || shape[3] == 0
                    || !bs[2].is_multiple_of(shape[2])
                    || !bs[3].is_multiple_of(shape[3])
                {
                    return Err(invalid("CPU matmul shapes differ"));
                }
                shape = [shape[1], bs[1], bs[2], bs[3]];
                kind = 0;
                sources.push(b);
                29
            }
            "ggml_mul_mat_id" => {
                let b = tensor(args, 2)?;
                let ids = tensor(args, 3)?;
                let bs = self.info(b)?.shape;
                let indices = self.info(ids)?;
                let is = indices.shape;
                if shape[3] != 1
                    || bs[3] != 1
                    || is[2] != 1
                    || is[3] != 1
                    || is[1] != bs[2]
                    || shape[0] != bs[0]
                    || bs[1] == 0
                    || !is[0].is_multiple_of(bs[1])
                    || indices.kind != 26
                {
                    return Err(invalid("CPU expert matmul geometry differs"));
                }
                shape = [shape[1], is[0], bs[2], 1];
                kind = 0;
                sources.push(b);
                sources.push(ids);
                30
            }
            "ggml_out_prod" => {
                let b = tensor(args, 2)?;
                let bs = self.info(b)?.shape;
                if shape[1] != bs[1] {
                    return Err(invalid("CPU outer product shapes differ"));
                }
                shape = [shape[0], bs[0], bs[2], bs[3]];
                kind = 0;
                sources.push(b);
                31
            }
            "ggml_scale" | "ggml_scale_bias" | "ggml_scale_bias_inplace" => {
                put_float(&mut p, 0, float(args, 2)?);
                if name != "ggml_scale" {
                    put_float(&mut p, 1, float(args, 3)?);
                }
                32
            }
            "ggml_acc" | "ggml_set" | "ggml_set_inplace" => {
                sources.push(tensor(args, 2)?);
                for i in 0..4 {
                    put_int(&mut p, i, i32::try_from(int(args, i + 3)?)?);
                }
                put_int(&mut p, 4, i32::from(inplace));
                if name == "ggml_acc" { 5 } else { 33 }
            }
            "ggml_cpy" => {
                let b = tensor(args, 2)?;
                let bi = self.info(b)?.clone();
                if product(shape)? != product(bi.shape)? {
                    return Err(invalid("CPU copy element counts differ"));
                }
                shape = bi.shape;
                kind = bi.kind;
                view = Some((b, bi.strides, 0));
                sources.push(b);
                34
            }
            "ggml_cont" => 35,
            "ggml_reshape_2d" | "ggml_reshape_3d" | "ggml_reshape_4d" => {
                shape = [1; 4];
                for (axis, dim) in shape.iter_mut().enumerate().take(args.len() - 2) {
                    *dim = usize_arg(args, axis + 2)?;
                }
                if !a_info.contiguous() || product(shape)? != a_info.elements() {
                    return Err(invalid(
                        "CPU reshape is not contiguous or changes element count",
                    ));
                }
                view = Some((a, super::contiguous_strides(kind, shape)?, 0));
                36
            }
            "ggml_view_tensor" => {
                view = Some((a, a_info.strides, 0));
                37
            }
            "ggml_view_1d" | "ggml_view_2d" | "ggml_view_3d" | "ggml_view_4d" => {
                let dims = match name {
                    "ggml_view_1d" => 1,
                    "ggml_view_2d" => 2,
                    "ggml_view_3d" => 3,
                    _ => 4,
                };
                shape = [1; 4];
                for (axis, dim) in shape.iter_mut().enumerate().take(dims) {
                    *dim = usize_arg(args, axis + 2)?;
                }
                let mut strides = super::contiguous_strides(kind, shape)?;
                for (axis, stride) in strides.iter_mut().enumerate().take(dims).skip(1) {
                    *stride = usize_arg(args, 1 + dims + axis)?;
                }
                for axis in dims..4 {
                    strides[axis] = strides[axis - 1]
                        .checked_mul(shape[axis - 1])
                        .ok_or_else(|| invalid("CPU view trailing stride overflow"))?;
                }
                let offset = usize_arg(args, args.len() - 1)?;
                p[..8].copy_from_slice(&(offset as u64).to_le_bytes());
                view = Some((a, strides, offset));
                37
            }
            "ggml_permute" => {
                let mut strides = [0; 4];
                let mut output = [0; 4];
                let mut seen = [false; 4];
                for axis in 0..4 {
                    let destination = usize_arg(args, axis + 2)?;
                    if destination >= 4 || seen[destination] {
                        return Err(invalid("CPU permutation axes differ"));
                    }
                    seen[destination] = true;
                    put_int(&mut p, axis, destination as i32);
                    output[destination] = shape[axis];
                    strides[destination] = a_info.strides[axis];
                }
                shape = output;
                view = Some((a, strides, 0));
                38
            }
            "ggml_transpose" => {
                shape.swap(0, 1);
                let mut strides = a_info.strides;
                strides.swap(0, 1);
                view = Some((a, strides, 0));
                39
            }
            "ggml_get_rows" => {
                let b = tensor(args, 2)?;
                let bs = self.info(b)?.shape;
                if bs[3] != 1 || shape[2] != bs[1] || shape[3] != bs[2] {
                    return Err(invalid("CPU gather batch shapes differ"));
                }
                shape = [shape[0], bs[0], bs[1], bs[2]];
                kind = if a_info.kind == 26 { 26 } else { 0 };
                sources.push(b);
                40
            }
            "ggml_get_rows_back" => {
                sources.push(tensor(args, 2)?);
                shape = self.info(tensor(args, 3)?)?.shape;
                kind = 0;
                41
            }
            "ggml_set_rows" => {
                let b = tensor(args, 2)?;
                let bi = self.info(b)?.clone();
                let ids = tensor(args, 3)?;
                let index_shape = self.info(ids)?.shape;
                if shape[0] != bi.shape[0]
                    || shape[2..] != bi.shape[2..]
                    || bi.shape[1] != index_shape[0]
                    || !bi.shape[2].is_multiple_of(index_shape[1])
                    || !bi.shape[3].is_multiple_of(index_shape[2])
                {
                    return Err(invalid("CPU row assignment shapes differ"));
                }
                sources = vec![b, ids, a];
                view = Some((a, a_info.strides, 0));
                42
            }
            "ggml_diag" => {
                shape[1] = shape[0];
                43
            }
            "ggml_diag_mask_inf" => {
                put_int(&mut p, 0, i32::try_from(int(args, 2)?)?);
                44
            }
            "ggml_soft_max" | "ggml_soft_max_ext" | "ggml_soft_max_ext_inplace" => {
                put_float(&mut p, 0, 1.0);
                if name != "ggml_soft_max" {
                    if let Argument::Tensor(mask) = at(args, 2)? {
                        sources.push(*mask);
                    } else if !matches!(at(args, 2)?, Argument::Null) {
                        return Err(invalid("CPU softmax mask differs"));
                    }
                    put_float(&mut p, 0, float(args, 3)?);
                    put_float(&mut p, 1, float(args, 4)?);
                }
                46
            }
            "ggml_soft_max_ext_back" => {
                sources.push(tensor(args, 2)?);
                put_float(&mut p, 0, float(args, 3)?);
                put_float(&mut p, 1, float(args, 4)?);
                47
            }
            "ggml_flash_attn_ext" => {
                let keys = tensor(args, 2)?;
                let values = tensor(args, 3)?;
                let ks = self.info(keys)?.shape;
                let vs = self.info(values)?.shape;
                if shape[0] != ks[0]
                    || shape[3] != ks[3]
                    || shape[3] != vs[3]
                    || ks[1] != vs[1]
                    || ks[2] == 0
                    || vs[2] == 0
                    || !shape[2].is_multiple_of(ks[2])
                    || !shape[2].is_multiple_of(vs[2])
                {
                    return Err(invalid("CPU attention input geometry differs"));
                }
                sources.push(keys);
                sources.push(values);
                if let Argument::Tensor(mask) = at(args, 4)? {
                    let mi = self.info(*mask)?;
                    if mi.kind != 1
                        || mi.shape[0] < ks[1]
                        || mi.shape[1] < shape[1]
                        || !shape[2].is_multiple_of(mi.shape[2])
                        || !shape[3].is_multiple_of(mi.shape[3])
                    {
                        return Err(invalid("CPU attention mask geometry differs"));
                    }
                    sources.push(*mask);
                } else if !matches!(at(args, 4)?, Argument::Null) {
                    return Err(invalid("CPU attention mask argument differs"));
                }
                for index in 0..3 {
                    put_float(&mut p, index, float(args, index + 5)?);
                }
                shape = [vs[0], shape[2], shape[1], shape[3]];
                kind = 0;
                74
            }
            "ggml_clamp" => {
                put_float(&mut p, 0, float(args, 2)?);
                put_float(&mut p, 1, float(args, 3)?);
                50
            }
            "ggml_rope"
            | "ggml_rope_ext"
            | "ggml_rope_ext_inplace"
            | "ggml_rope_ext_back"
            | "ggml_rope_multi"
            | "ggml_rope_multi_inplace"
            | "ggml_rope_multi_back" => {
                sources.push(tensor(args, 2)?);
                let ordinary = name == "ggml_rope";
                let multi = name.contains("multi");
                let dimension_index = if ordinary { 3 } else { 4 };
                let dims = usize_arg(args, dimension_index)?;
                let mode_index = dimension_index + if multi { 2 } else { 1 };
                let mode = i32::try_from(int(args, mode_index)?)?;
                if dims == 0
                    || !dims.is_multiple_of(2)
                    || dims > shape[0]
                    || !matches!(mode, 0 | 2 | 8 | 24 | 40)
                {
                    return Err(invalid("CPU rotary dimensions or mode differ"));
                }
                put_int(&mut p, 1, i32::try_from(dims)?);
                put_int(&mut p, 2, mode);
                if ordinary {
                    put_float(&mut p, 5, 10000.0);
                    put_float(&mut p, 6, 1.0);
                    put_float(&mut p, 8, 1.0);
                } else {
                    if let Argument::Tensor(factors) = at(args, 3)? {
                        sources.push(*factors);
                    } else if !matches!(at(args, 3)?, Argument::Null) {
                        return Err(invalid("CPU rotary factor type differs"));
                    }
                    put_int(&mut p, 4, i32::try_from(int(args, mode_index + 1)?)?);
                    for index in 0..6 {
                        put_float(&mut p, 5 + index, float(args, mode_index + 2 + index)?);
                    }
                }
                if multi {
                    let Argument::Integers(sections) = at(args, dimension_index + 1)? else {
                        return Err(invalid("CPU rotary sections require an integer list"));
                    };
                    if sections.len() != 4 || sections.iter().any(|&x| x < 0) {
                        return Err(invalid("CPU rotary section extent differs"));
                    }
                    for (index, &section) in sections.iter().enumerate() {
                        put_int(&mut p, 11 + index, i32::try_from(section)?);
                    }
                }
                if name.ends_with("_back") { 49 } else { 48 }
            }
            "ggml_conv_transpose_1d" => {
                let b = tensor(args, 2)?;
                let bs = self.info(b)?.shape;
                let stride = usize_arg(args, 3)?;
                if stride == 0
                    || int(args, 4)? != 0
                    || int(args, 5)? != 1
                    || shape[2] != bs[1]
                    || shape[3] != 1
                    || bs[2] != 1
                    || bs[3] != 1
                {
                    return Err(invalid("CPU transposed convolution geometry differs"));
                }
                shape = [
                    (bs[0] - 1)
                        .checked_mul(stride)
                        .and_then(|x| x.checked_add(shape[0]))
                        .ok_or_else(|| invalid("CPU convolution extent overflow"))?,
                    shape[1],
                    1,
                    1,
                ];
                kind = 0;
                sources.push(b);
                put_int(&mut p, 0, i32::try_from(stride)?);
                51
            }
            "ggml_col2im_1d" => {
                let stride = usize_arg(args, 2)?;
                let channels = usize_arg(args, 3)?;
                let padding = usize_arg(args, 4)?;
                if stride == 0
                    || channels == 0
                    || !a_info.contiguous()
                    || shape[2] != 1
                    || shape[3] != 1
                    || !shape[0].is_multiple_of(channels)
                {
                    return Err(invalid("CPU col2im geometry differs"));
                }
                let extent = (shape[1] - 1)
                    .checked_mul(stride)
                    .and_then(|x| x.checked_add(shape[0] / channels))
                    .and_then(|x| x.checked_sub(2 * padding))
                    .ok_or_else(|| invalid("CPU col2im extent overflow"))?;
                if extent == 0 {
                    return Err(invalid("CPU col2im crop removes output"));
                }
                shape = [extent, channels, 1, 1];
                for (index, value) in [stride, channels, padding].into_iter().enumerate() {
                    put_int(&mut p, index, i32::try_from(value)?);
                }
                55
            }
            "ggml_im2col_back" => {
                let kernel = tensor(args, 2)?;
                if kind != 0 || !matches!(self.info(kernel)?.kind, 0 | 1) {
                    return Err(invalid("CPU im2col backward storage types differ"));
                }
                let Argument::Integers(dimensions) = at(args, 3)? else {
                    return Err(invalid("CPU im2col backward shape missing"));
                };
                if dimensions.len() != 4 {
                    return Err(invalid("CPU im2col backward rank differs"));
                }
                for axis in 0..4 {
                    shape[axis] = usize::try_from(dimensions[axis])?;
                }
                kind = 0;
                sources.push(kernel);
                for index in 0..6 {
                    put_int(&mut p, index, i32::try_from(int(args, index + 4)?)?);
                }
                put_int(&mut p, 6, i32::from(boolean(args, 10)?));
                53
            }
            "ggml_pool_2d_back" => {
                let forward = tensor(args, 2)?;
                if kind != 0 || self.info(forward)?.kind != 0 {
                    return Err(invalid("CPU pooling backward requires binary32 storage"));
                }
                shape = self.info(forward)?.shape;
                kind = 0;
                sources.push(forward);
                for index in 0..5 {
                    put_int(&mut p, index, i32::try_from(int(args, index + 3)?)?);
                }
                put_int(&mut p, 5, float(args, 8)? as i32);
                put_int(&mut p, 6, float(args, 9)? as i32);
                62
            }
            "ggml_im2col" | "ggml_conv_2d_direct" | "ggml_conv_2d_dw_direct" => {
                let b = tensor(args, 2)?;
                let bs = self.info(b)?.shape;
                let two_dimensional = if name == "ggml_im2col" {
                    boolean(args, 9)?
                } else {
                    true
                };
                for index in 0..6 {
                    put_int(&mut p, index, i32::try_from(int(args, index + 3)?)?);
                }
                let width = convolution_extent(
                    bs[0],
                    shape[0],
                    int(args, 3)?,
                    int(args, 5)?,
                    int(args, 7)?,
                )?;
                let height = if two_dimensional {
                    convolution_extent(
                        bs[1],
                        shape[1],
                        int(args, 4)?,
                        int(args, 6)?,
                        int(args, 8)?,
                    )?
                } else {
                    bs[2]
                };
                if name == "ggml_im2col" {
                    if (two_dimensional && shape[2] != bs[2])
                        || (!two_dimensional && (shape[1] != bs[1] || bs[3] != 1))
                    {
                        return Err(invalid("CPU im2col channel geometry differs"));
                    }
                    shape = [
                        shape[0] * shape[1] * if two_dimensional { shape[2] } else { 1 },
                        width,
                        height,
                        if two_dimensional { bs[3] } else { 1 },
                    ];
                    kind = i32::try_from(int(args, 10)?)?;
                    put_int(&mut p, 6, i32::from(two_dimensional));
                    sources.push(b);
                    52
                } else {
                    let depthwise = name == "ggml_conv_2d_dw_direct";
                    if depthwise && (shape[2] != 1 || shape[3] != bs[2])
                        || !depthwise && shape[2] != bs[2]
                    {
                        return Err(invalid("CPU 2D convolution channel geometry differs"));
                    }
                    shape = [width, height, shape[3], bs[3]];
                    kind = self.info(b)?.kind;
                    sources.push(b);
                    if depthwise { 58 } else { 56 }
                }
            }
            "ggml_conv_3d_direct" => {
                let b = tensor(args, 2)?;
                let bs = self.info(b)?.shape;
                let channels = usize_arg(args, 12)?;
                let batches = usize_arg(args, 13)?;
                let output_channels = usize_arg(args, 14)?;
                if shape[3] != channels * output_channels || bs[3] != channels * batches {
                    return Err(invalid("CPU 3D convolution channel geometry differs"));
                }
                for axis in 0..3 {
                    shape[axis] = convolution_extent(
                        bs[axis],
                        shape[axis],
                        int(args, 3 + axis)?,
                        int(args, 6 + axis)?,
                        int(args, 9 + axis)?,
                    )?;
                }
                shape[3] = output_channels * batches;
                kind = 0;
                sources.push(b);
                for index in 0..12 {
                    put_int(&mut p, index, i32::try_from(int(args, 3 + index)?)?);
                }
                57
            }
            "ggml_im2col_3d" => {
                let image = tensor(args, 2)?;
                let image_shape = self.info(image)?.shape;
                let channels = usize_arg(args, 3)?;
                if channels == 0 {
                    return Err(invalid("CPU 3D im2col channel geometry differs"));
                }
                // Native IM2COL_3D counts complete channel groups with integer
                // division. A trailing group is ignored; fewer than IC planes
                // produces an empty output, including source strided views.
                let batches = image_shape[3] / channels;
                let mut spatial = [0; 3];
                for axis in 0..3 {
                    spatial[axis] = convolution_extent(
                        image_shape[axis],
                        shape[axis],
                        int(args, 4 + axis)?,
                        int(args, 7 + axis)?,
                        int(args, 10 + axis)?,
                    )?;
                }
                shape = [
                    product([shape[0], shape[1], shape[2], channels])?,
                    spatial[0],
                    spatial[1],
                    spatial[2]
                        .checked_mul(batches)
                        .ok_or_else(|| invalid("CPU 3D im2col batch extent overflow"))?,
                ];
                sources.push(image);
                kind = i32::try_from(int(args, 13)?)?;
                for index in 0..9 {
                    put_int(&mut p, index, i32::try_from(int(args, 4 + index)?)?);
                }
                put_int(&mut p, 9, i32::try_from(channels)?);
                54
            }
            "ggml_conv_transpose_2d_p0" => {
                let b = tensor(args, 2)?;
                let bs = self.info(b)?.shape;
                let stride = usize_arg(args, 3)?;
                if stride == 0 || shape[3] != bs[2] {
                    return Err(invalid("CPU 2D transposed convolution geometry differs"));
                }
                shape = [
                    (bs[0] - 1) * stride + shape[0],
                    (bs[1] - 1) * stride + shape[1],
                    shape[2],
                    bs[3],
                ];
                kind = 0;
                sources.push(b);
                put_int(&mut p, 0, i32::try_from(stride)?);
                59
            }
            "ggml_pool_1d" | "ggml_pool_2d" => {
                let two_dimensional = name == "ggml_pool_2d";
                let pool = int(args, 2)?;
                if !(0..=1).contains(&pool) {
                    return Err(invalid("CPU pool operation differs"));
                }
                put_int(&mut p, 0, i32::try_from(pool)?);
                let kernel_width = int(args, 3)?;
                let kernel_height = if two_dimensional { int(args, 4)? } else { 1 };
                let stride_width = int(args, if two_dimensional { 5 } else { 4 })?;
                let stride_height = if two_dimensional { int(args, 6)? } else { 1 };
                let padding_width = if two_dimensional {
                    float(args, 7)?
                } else {
                    int(args, 5)? as f32
                };
                let padding_height = if two_dimensional {
                    float(args, 8)?
                } else {
                    0.0
                };
                if kernel_width <= 0
                    || kernel_height <= 0
                    || stride_width <= 0
                    || stride_height <= 0
                {
                    return Err(invalid("CPU pool kernel or stride differs"));
                }
                let width = ((shape[0] as f32 + 2.0 * padding_width - kernel_width as f32)
                    / stride_width as f32
                    + 1.0) as i64;
                let height = if two_dimensional {
                    ((shape[1] as f32 + 2.0 * padding_height - kernel_height as f32)
                        / stride_height as f32
                        + 1.0) as i64
                } else {
                    shape[1] as i64
                };
                if width <= 0 || height <= 0 {
                    return Err(invalid("CPU pool output extent differs"));
                }
                shape[0] = usize::try_from(width)?;
                shape[1] = usize::try_from(height)?;
                kind = 0;
                if two_dimensional {
                    for (index, value) in [
                        kernel_width,
                        kernel_height,
                        stride_width,
                        stride_height,
                        padding_width as i64,
                        padding_height as i64,
                    ]
                    .into_iter()
                    .enumerate()
                    {
                        put_int(&mut p, index + 1, i32::try_from(value)?);
                    }
                    61
                } else {
                    for (index, value) in [kernel_width, stride_width, padding_width as i64]
                        .into_iter()
                        .enumerate()
                    {
                        put_int(&mut p, index + 1, i32::try_from(value)?);
                    }
                    60
                }
            }
            "ggml_upscale" | "ggml_interpolate" => {
                let mode = if name == "ggml_upscale" {
                    let factor = usize_arg(args, 2)?;
                    if factor <= 1 {
                        return Err(invalid("CPU upscale factor must exceed one"));
                    }
                    shape[0] *= factor;
                    shape[1] *= factor;
                    int(args, 3)?
                } else {
                    for (axis, dimension) in shape.iter_mut().enumerate() {
                        *dimension = usize_arg(args, axis + 2)?;
                    }
                    int(args, 6)?
                };
                if !(0..=2).contains(&(mode & 255))
                    || mode & 512 != 0 && mode & 255 != 1
                    || shape.contains(&0)
                {
                    return Err(invalid("CPU interpolation mode or extent differs"));
                }
                put_int(&mut p, 0, i32::try_from(mode)?);
                63
            }
            "ggml_pad_reflect_1d" => {
                let left = usize_arg(args, 2)?;
                let right = usize_arg(args, 3)?;
                if left >= shape[0] || right >= shape[0] {
                    return Err(invalid("CPU reflection padding exceeds input width"));
                }
                shape[0] += left + right;
                put_int(&mut p, 0, i32::try_from(left)?);
                put_int(&mut p, 1, i32::try_from(right)?);
                65
            }
            "ggml_pad" | "ggml_pad_circular" | "ggml_pad_ext" | "ggml_pad_ext_circular" => {
                for (axis, dimension) in shape.iter_mut().enumerate() {
                    let (left, right) = if name.contains("_ext") {
                        (int(args, 2 + 2 * axis)?, int(args, 3 + 2 * axis)?)
                    } else {
                        (0, int(args, 2 + axis)?)
                    };
                    let extent = *dimension as i64 + left + right;
                    if extent < 0 {
                        return Err(invalid("CPU padding creates negative extent"));
                    }
                    *dimension = usize::try_from(extent)?;
                    put_int(&mut p, 2 * axis, i32::try_from(left)?);
                    put_int(&mut p, 2 * axis + 1, i32::try_from(right)?);
                }
                put_int(&mut p, 8, i32::from(name.ends_with("_circular")));
                64
            }
            "ggml_roll" => {
                for axis in 0..4 {
                    put_int(&mut p, axis, i32::try_from(int(args, axis + 2)?)?);
                }
                66
            }
            "ggml_timestep_embedding" => {
                let dim = usize_arg(args, 2)?;
                shape = [dim.next_multiple_of(2), a_info.shape[0], 1, 1];
                kind = 0;
                put_int(&mut p, 0, i32::try_from(dim)?);
                put_int(&mut p, 1, i32::try_from(int(args, 3)?)?);
                68
            }
            "ggml_argsort" | "ggml_argsort_top_k" | "ggml_top_k" => {
                kind = 26;
                put_int(
                    &mut p,
                    0,
                    if name == "ggml_argsort" {
                        i32::try_from(int(args, 2)?)?
                    } else {
                        1
                    },
                );
                if name != "ggml_argsort" {
                    let k = usize_arg(args, 2)?;
                    if k == 0 || k > shape[0] {
                        return Err(invalid("CPU top-k exceeds row width"));
                    }
                    shape[0] = k;
                }
                if name == "ggml_top_k" { 70 } else { 69 }
            }
            "ggml_leaky_relu" => {
                put_float(&mut p, 0, float(args, 2)?);
                inplace = boolean(args, 3)?;
                71
            }
            "ggml_tri" => {
                put_int(&mut p, 0, i32::try_from(int(args, 2)?)?);
                72
            }
            "ggml_fill" => {
                put_float(&mut p, 0, float(args, 2)?);
                73
            }
            "ggml_solve_tri" => {
                sources.push(tensor(args, 2)?);
                shape = self.info(sources[1])?.shape;
                if !boolean(args, 3)?
                    || !boolean(args, 4)?
                    || boolean(args, 5)?
                    || a_info.shape[0] != a_info.shape[1]
                    || a_info.shape[1..] != shape[1..]
                {
                    return Err(invalid("CPU triangular solve geometry differs"));
                }
                85
            }
            "ggml_ssm_conv" => {
                let weights = tensor(args, 2)?;
                let ws = self.info(weights)?.shape;
                if ws[0] == 0
                    || ws[1] != shape[1]
                    || ws[2] != 1
                    || ws[3] != 1
                    || shape[3] != 1
                    || shape[0] + 1 < ws[0]
                {
                    return Err(invalid("CPU SSM convolution geometry differs"));
                }
                shape = [shape[1], shape[0] + 1 - ws[0], shape[2], 1];
                kind = 0;
                sources.push(weights);
                76
            }
            "ggml_ssm_scan" => {
                for index in 2..=7 {
                    sources.push(tensor(args, index)?);
                }
                let xs = self.info(sources[1])?.shape;
                let ds = self.info(sources[2])?.shape;
                let decay = self.info(sources[3])?.shape;
                let bs = self.info(sources[4])?.shape;
                let cs = self.info(sources[5])?.shape;
                let ids = self.info(sources[6])?;
                let snapshots = int(args, 8)?;
                if snapshots < 1
                    || shape[1] != xs[0]
                    || shape[2] != xs[1]
                    || ds != [xs[1], xs[2], xs[3], 1]
                    || (decay[0] != 1 && (decay[0] != shape[0] || snapshots != 1))
                    || decay[1..] != [xs[1], 1, 1]
                    || bs != cs
                    || bs[0] != shape[0]
                    || bs[2..] != xs[2..]
                    || bs[1] == 0
                    || !xs[1].is_multiple_of(bs[1])
                    || ids.shape != [xs[3], 1, 1, 1]
                    || ids.kind != 26
                    || sources[..6]
                        .iter()
                        .any(|&id| self.info(id).is_ok_and(|info| info.kind != 0))
                {
                    return Err(invalid("CPU state scan geometry differs"));
                }
                shape = [
                    product(xs)?
                        + usize::try_from(snapshots)? * shape[0] * shape[1] * shape[2] * xs[3],
                    1,
                    1,
                    1,
                ];
                kind = 0;
                put_int(&mut p, 0, i32::try_from(snapshots)?);
                77
            }
            "ggml_rwkv_wkv6" | "ggml_gated_linear_attn" | "ggml_rwkv_wkv7" => {
                let operation = match name {
                    "ggml_rwkv_wkv6" => 82,
                    "ggml_gated_linear_attn" => 83,
                    _ => 84,
                };
                let state_argument = match operation {
                    82 => 6,
                    83 => 5,
                    _ => 7,
                };
                for index in 2..=state_argument {
                    sources.push(tensor(args, index)?);
                }
                let state_info = self.info(sources[state_argument - 1])?;
                let sequences = state_info.shape[1];
                let width = shape[0];
                let heads = shape[1];
                let tokens = shape[2];
                if sequences == 0
                    || tokens == 0
                    || !tokens.is_multiple_of(sequences)
                    || state_info.elements() != width * width * heads * sequences
                    || sources
                        .iter()
                        .any(|&id| self.info(id).is_ok_and(|info| info.kind != 0))
                {
                    return Err(invalid("CPU recurrent attention state geometry differs"));
                }
                for (index, &source) in sources.iter().enumerate().take(state_argument - 1) {
                    let actual = self.info(source)?.shape;
                    if operation == 82 && index == 3 {
                        if self.info(source)?.elements() < width * heads {
                            return Err(invalid("CPU WKV time weight extent differs"));
                        }
                    } else if actual != shape {
                        return Err(invalid("CPU recurrent attention vector geometry differs"));
                    }
                }
                if operation == 83 {
                    put_float(&mut p, 0, float(args, 6)?);
                }
                shape = [width * heads, tokens + width * sequences, 1, 1];
                kind = 0;
                operation
            }
            "ggml_dsv4_hc_comb" => {
                let scale = tensor(args, 2)?;
                let base = tensor(args, 3)?;
                let scales = self.info(scale)?;
                let bases = self.info(base)?;
                let iterations = int(args, 5)?;
                if shape[0] != 24
                    || shape[2..] != [1, 1]
                    || scales.shape[0] < 3
                    || scales.shape[1..] != [1, 1, 1]
                    || bases.shape != [24, 1, 1, 1]
                    || iterations <= 0
                    || [a, scale, base]
                        .iter()
                        .any(|&id| self.info(id).is_ok_and(|info| info.kind != 0))
                {
                    return Err(invalid("CPU hyperconnection mixture geometry differs"));
                }
                sources.push(scale);
                sources.push(base);
                shape = [4, 4, shape[1], 1];
                kind = 0;
                put_float(&mut p, 0, float(args, 4)?);
                put_int(&mut p, 1, i32::try_from(iterations)?);
                88
            }
            "ggml_gated_delta_net"
            | "ggml_gated_delta_net_rows"
            | "ggml_gated_delta_net_replay" => {
                for index in 2..=6 {
                    sources.push(tensor(args, index)?);
                }
                let rows = name.ends_with("_rows");
                let replay = name.ends_with("_replay");
                let vs = self.info(sources[2])?.shape;
                let qs = shape;
                let ks = self.info(sources[1])?.shape;
                let gs = self.info(sources[3])?.shape;
                let bs = self.info(sources[4])?.shape;
                let ss = self.info(sources[5])?.shape;
                let [width, heads, tokens, sequences] = vs;
                let snapshots = int(
                    args,
                    if replay {
                        9
                    } else if rows {
                        8
                    } else {
                        7
                    },
                )?;
                if width == 0
                    || heads == 0
                    || sequences == 0
                    || snapshots < 1
                    || qs[0] != width
                    || ks[0] != width
                    || qs[1] == 0
                    || ks[1] == 0
                    || !heads.is_multiple_of(qs[1])
                    || !heads.is_multiple_of(ks[1])
                    || qs[2] != tokens
                    || ks[2] != tokens
                    || qs[3] == 0
                    || ks[3] == 0
                    || !sequences.is_multiple_of(qs[3])
                    || !sequences.is_multiple_of(ks[3])
                    || (gs[0] != 1 && gs[0] != width)
                    || gs[1..] != vs[1..]
                    || bs != [1, heads, tokens, sequences]
                    || sources
                        .iter()
                        .any(|&id| self.info(id).is_ok_and(|info| info.kind != 0))
                {
                    return Err(invalid("CPU gated delta input geometry differs"));
                }
                if rows {
                    let indices = tensor(args, 7)?;
                    let ii = self.info(indices)?;
                    if ss[0] != width * width * heads
                        || ss[2..] != [1, 1]
                        || ii.shape != [sequences, 1, 1, 1]
                        || ii.kind != 26
                    {
                        return Err(invalid("CPU gated delta cache row geometry differs"));
                    }
                    sources.push(indices);
                } else if ss != [width, width, heads, sequences] {
                    return Err(invalid("CPU gated delta state geometry differs"));
                }
                put_int(&mut p, 0, i32::try_from(snapshots)?);
                kind = 0;
                if replay {
                    let records = tensor(args, 7)?;
                    let control = tensor(args, 8)?;
                    let record_width = heads * (2 * width + 1);
                    let ri = self.info(records)?;
                    let ci = self.info(control)?;
                    if snapshots < 2
                        || tokens == 0
                        || gs[0] != 1
                        || ri.kind != 0
                        || ri.shape
                            != [
                                record_width * (usize::try_from(snapshots)? - 1),
                                2,
                                sequences,
                                1,
                            ]
                        || ci.kind != 26
                        || ci.elements() != 2 * sequences
                    {
                        return Err(invalid("CPU gated delta replay record geometry differs"));
                    }
                    sources.extend([records, control]);
                    shape = [
                        width * heads * tokens * sequences
                            + width * width * heads * sequences
                            + (usize::try_from(snapshots)? - 1) * record_width * sequences,
                        1,
                        1,
                        1,
                    ];
                    101
                } else {
                    shape = [
                        width * heads,
                        tokens * sequences + usize::try_from(snapshots)? * width * sequences,
                        1,
                        1,
                    ];
                    86
                }
            }
            "ggml_dsv4_hc_pre" => {
                let weights = tensor(args, 2)?;
                let wi = self.info(weights)?;
                if shape[1] == 0
                    || shape[3] != 1
                    || wi.shape != [shape[1], shape[2], 1, 1]
                    || wi.kind != 0
                    || kind != 0
                {
                    return Err(invalid(
                        "CPU hyperconnection pre-projection geometry differs",
                    ));
                }
                sources.push(weights);
                shape = [shape[0], shape[2], 1, 1];
                89
            }
            "ggml_lightning_indexer" => {
                let keys = tensor(args, 2)?;
                let weights = tensor(args, 3)?;
                let mask = tensor(args, 4)?;
                let ki = self.info(keys)?;
                let wi = self.info(weights)?;
                let mi = self.info(mask)?;
                if kind != 0
                    || wi.kind != 0
                    || mi.kind != 1
                    || shape[0] != ki.shape[0]
                    || ki.shape[1] != 1
                    || shape[1] != wi.shape[0]
                    || shape[2] != wi.shape[1]
                    || wi.shape[2] != 1
                    || mi.shape[0] != ki.shape[2]
                    || mi.shape[1] != shape[2]
                    || mi.shape[2] != 1
                    || shape[3] != ki.shape[3]
                    || shape[3] != wi.shape[3]
                    || mi.shape[3] == 0
                    || !shape[3].is_multiple_of(mi.shape[3])
                {
                    return Err(invalid("CPU lightning indexer geometry differs"));
                }
                shape = [ki.shape[2], shape[2], 1, shape[3]];
                sources.extend([keys, weights, mask]);
                87
            }
            "ggml_dsv4_hc_post" => {
                let residual = tensor(args, 2)?;
                let post = tensor(args, 3)?;
                let combination = tensor(args, 4)?;
                let rs = self.info(residual)?.shape;
                let ps = self.info(post)?.shape;
                let cs = self.info(combination)?.shape;
                if shape[2..] != [1, 1]
                    || rs[0] != shape[0]
                    || rs[1] == 0
                    || rs[2..] != [shape[1], 1]
                    || ps != [rs[1], shape[1], 1, 1]
                    || cs != [rs[1], rs[1], shape[1], 1]
                    || [a, residual, post, combination]
                        .iter()
                        .any(|&id| self.info(id).is_ok_and(|info| info.kind != 0))
                {
                    return Err(invalid(
                        "CPU hyperconnection post-projection geometry differs",
                    ));
                }
                sources.extend([residual, post, combination]);
                shape = [shape[0], rs[1], shape[1], 1];
                90
            }
            "ggml_opt_step_adamw" | "ggml_opt_step_sgd" => {
                let adam = name == "ggml_opt_step_adamw";
                if a_info.flags & 16 == 0 || kind != 0 {
                    return Err(invalid("CPU optimizer requires an F32 parameter tensor"));
                }
                for index in 2..if adam { 5 } else { 3 } {
                    let input = tensor(args, index)?;
                    if self.info(input)?.shape != shape {
                        return Err(invalid("CPU optimizer moment/gradient shape differs"));
                    }
                    sources.push(input);
                }
                let params = tensor(args, if adam { 5 } else { 3 })?;
                let pi = self.info(params)?;
                if pi.kind != 0 || pi.elements() != if adam { 7 } else { 2 } {
                    return Err(invalid("CPU optimizer parameter extent differs"));
                }
                sources.push(params);
                view = Some((a, a_info.strides, 0));
                if adam { 98 } else { 99 }
            }
            "ggml_unary" | "ggml_relu" | "ggml_silu" | "ggml_sigmoid" | "ggml_tanh"
            | "ggml_softplus" | "ggml_floor" | "ggml_ceil" | "ggml_round" | "ggml_trunc"
            | "ggml_xielu" => {
                let unary = match name {
                    "ggml_relu" => 6,
                    "ggml_silu" => 10,
                    "ggml_sigmoid" => 7,
                    "ggml_tanh" => 4,
                    "ggml_softplus" => 15,
                    "ggml_xielu" => 17,
                    "ggml_floor" => 18,
                    "ggml_ceil" => 19,
                    "ggml_round" => 20,
                    "ggml_trunc" => 21,
                    _ => i32::try_from(int(args, 2)?)?,
                };
                if !(0..22).contains(&unary) {
                    return Err(invalid("unknown CPU unary discriminator"));
                }
                put_int(&mut p, 0, unary);
                if name == "ggml_xielu" {
                    #[allow(
                        clippy::imprecise_flops,
                        reason = "The pinned softplus parameter transformation rounds the addition before logf."
                    )]
                    let softplus = |x: f32| if x > 20.0 { x } else { (1.0 + x.exp()).ln() };
                    let beta = float(args, 4)?;
                    put_float(&mut p, 1, beta + softplus(float(args, 2)?));
                    put_float(&mut p, 2, softplus(float(args, 3)?));
                    put_float(&mut p, 3, beta);
                    put_float(&mut p, 4, float(args, 5)?);
                }
                91
            }
            "ggml_cross_entropy_loss" | "ggml_cross_entropy_loss_back" => {
                sources.push(tensor(args, 2)?);
                if name.ends_with("_back") {
                    sources.push(tensor(args, 3)?);
                    shape = self.info(sources[1])?.shape;
                    if a_info.elements() != 1 || shape != self.info(sources[2])?.shape {
                        return Err(invalid("CPU cross-entropy backward inputs differ"));
                    }
                    97
                } else {
                    shape = [1; 4];
                    96
                }
            }
            "ggml_glu" | "ggml_glu_split" | "ggml_swiglu_split" | "ggml_swiglu_oai" => {
                let glu = if name == "ggml_glu" {
                    let glu = i32::try_from(int(args, 2)?)?;
                    // The pinned constructor and CPU kernels use floor(n/2),
                    // leaving an odd row's final element unused.
                    shape[0] /= 2;
                    put_int(&mut p, 1, i32::from(boolean(args, 3)?));
                    glu
                } else {
                    let b = tensor(args, 2)?;
                    if self.info(b)?.shape != shape {
                        return Err(invalid("CPU GLU shapes differ"));
                    }
                    sources.push(b);
                    match name {
                        "ggml_swiglu_split" => 2,
                        "ggml_swiglu_oai" => {
                            put_float(&mut p, 2, float(args, 3)?);
                            put_float(&mut p, 3, float(args, 4)?);
                            3
                        }
                        _ => i32::try_from(int(args, 3)?)?,
                    }
                };
                put_int(&mut p, 0, glu);
                100
            }
            _ => {
                return Err(invalid(format!(
                    "unsupported CPU production graph call {name}"
                )));
            }
        };
        if matches!(op, 2 | 6..=8) {
            let b = tensor(args, 2)?;
            let bs = self.info(b)?.shape;
            if bs
                .iter()
                .zip(shape)
                .any(|(&b, a)| b == 0 || !a.is_multiple_of(b))
            {
                return Err(invalid("CPU binary shapes cannot broadcast"));
            }
            sources.push(b);
        }
        if inplace {
            view = Some((a, a_info.strides, 0));
        }
        let channel_contiguous = if op == 58 {
            let input = self.info(sources[1])?;
            let width = type_layout(input.kind)?.bytes;
            input.strides[2] == width
                && input.strides[0] == input.shape[2] * width
                && input.strides[1] == input.shape[0] * input.shape[2] * width
        } else {
            false
        };
        let observation = self.derived(ctx, kind, shape, op, sources, p, view)?;
        if channel_contiguous && let Observation::Tensor(Some(id)) = observation {
            let info = self.info_mut(id)?;
            let width = type_layout(kind)?.bytes;
            info.strides[0] = shape[2] * width;
            info.strides[1] = shape[0] * shape[2] * width;
            info.strides[2] = width;
        }
        Ok(observation)
    }
}
