//! Clean-room builders and metadata dispatch for all 173 allowlisted `ggml_*` intrinsics.
//!
//! No `bonsai_prism_runtime`, `bonsai_gguf`, or native GGML libraries are linked or called.

use std::collections::BTreeSet;

use serde_json::Value;

use super::eval_ops;
use super::state::{
    BackendState, GraphRecord, ScalarValue, alloc_tensor, arg_bool, arg_f32, arg_handle, arg_i32,
    arg_i32_array, arg_i64, arg_i64_array, arg_opt_tensor, arg_str, arg_tensor_array, arg_u64,
    set_tensor_zero,
};
use super::types::{
    FLAG_LOSS, FLAG_PARAM, GGML_MAX_DIMS, OP_ACC, OP_ADD, OP_ADD_ID, OP_ARANGE, OP_ARGMAX,
    OP_ARGSORT, OP_CLAMP, OP_COL2IM_1D, OP_CONCAT, OP_CONT, OP_CONV_2D, OP_CONV_2D_DW, OP_CONV_3D,
    OP_CONV_TRANSPOSE_1D, OP_CONV_TRANSPOSE_2D, OP_COS, OP_COUNT_EQUAL, OP_CPY,
    OP_CROSS_ENTROPY_LOSS, OP_CROSS_ENTROPY_LOSS_BACK, OP_CUMSUM, OP_DIAG, OP_DIAG_MASK_INF,
    OP_DIV, OP_DSV4_HC_COMB, OP_DSV4_HC_POST, OP_DSV4_HC_PRE, OP_DUP, OP_FILL, OP_FLASH_ATTN_EXT,
    OP_GATED_DELTA_NET, OP_GATED_DELTA_NET_REPLAY, OP_GATED_LINEAR_ATTN, OP_GET_ROWS,
    OP_GET_ROWS_BACK, OP_GLU, OP_GROUP_NORM, OP_IM2COL, OP_IM2COL_3D, OP_L2_NORM, OP_LEAKY_RELU,
    OP_LIGHTNING_INDEXER, OP_LOG, OP_MEAN, OP_MUL, OP_MUL_MAT, OP_MUL_MAT_ID, OP_NONE, OP_NORM,
    OP_OPT_STEP_ADAMW, OP_OPT_STEP_SGD, OP_OUT_PROD, OP_PAD, OP_PAD_REFLECT_1D, OP_PERMUTE,
    OP_POOL_1D, OP_POOL_2D, OP_REPEAT, OP_REPEAT_BACK, OP_RESHAPE, OP_RMS_NORM, OP_RMS_NORM_BACK,
    OP_ROLL, OP_ROPE, OP_ROPE_BACK, OP_RWKV_WKV6, OP_RWKV_WKV7, OP_SCALE, OP_SET, OP_SET_ROWS,
    OP_SILU_BACK, OP_SIN, OP_SOFT_MAX, OP_SOFT_MAX_BACK, OP_SOLVE_TRI, OP_SQR, OP_SQRT,
    OP_SSM_CONV, OP_SSM_SCAN, OP_SUB, OP_SUM, OP_SUM_ROWS, OP_TIMESTEP_EMBEDDING, OP_TOP_K,
    OP_TRANSPOSE, OP_TRI, OP_UNARY, OP_UPSCALE, OP_VIEW, STATUS_SUCCESS, TYPE_F32, TYPE_I32,
    TYPE_I64, TensorMeta, UNARY_OP_CEIL, UNARY_OP_FLOOR, UNARY_OP_RELU, UNARY_OP_ROUND,
    UNARY_OP_SIGMOID, UNARY_OP_SILU, UNARY_OP_SOFTPLUS, UNARY_OP_TANH, UNARY_OP_TRUNC,
    UNARY_OP_XIELU, blck_size, f32_to_i64_trunc, f64_to_f32, glu_op_name, is_quantized, op_name,
    type_name, type_size, unary_op_name,
};
use crate::{Result, invalid};

fn t_id(args: &[Value], i: usize) -> Result<u32> {
    arg_handle(&args[i], "Tensor")
}

fn t_meta(state: &BackendState, args: &[Value], i: usize) -> Result<TensorMeta> {
    Ok(state.tensor(t_id(args, i)?)?.clone())
}

fn res_tensor(res: &Value) -> Result<u32> {
    arg_handle(res, "Tensor")
}

fn res_graph(res: &Value) -> Result<u32> {
    arg_handle(res, "Graph")
}

fn store_scalar(state: &mut BackendState, res: &Value, value: ScalarValue) -> Result<()> {
    let sid = res
        .get("Scalar")
        .and_then(Value::as_u64)
        .ok_or_else(|| invalid("call missing Result.Scalar"))?;
    state.scalars.insert(sid, value);
    Ok(())
}

fn finish_op(
    state: &mut BackendState,
    ctx_id: u32,
    new_id: u32,
    kind: i32,
    dims: [i64; GGML_MAX_DIMS],
    op_code: i32,
    sources: &[Option<u32>],
) -> Result<()> {
    let mut meta = alloc_tensor(state, ctx_id, new_id, kind, dims)?;
    meta.op = op_code;
    for (slot, maybe_src) in meta.sources.iter_mut().zip(sources.iter()) {
        *slot = maybe_src.map(|id| state.canonical_id(id)).transpose()?;
    }
    state.tensors.insert(new_id, meta);
    Ok(())
}

fn finish_inplace(
    state: &mut BackendState,
    ctx_id: u32,
    new_id: u32,
    target_id: u32,
    op_code: i32,
    sources: &[Option<u32>],
) -> Result<()> {
    let base = state.tensor(target_id)?.clone();
    let mut meta = alloc_tensor(state, ctx_id, new_id, base.kind, base.ne)?;
    meta.nb = base.nb;
    meta.bytes = base.bytes;
    meta.op = op_code;
    meta.view_source = Some(base.view_source.unwrap_or(base.id));
    meta.view_offset = base.view_offset;
    for (slot, maybe_src) in meta.sources.iter_mut().zip(sources.iter()) {
        *slot = maybe_src.map(|id| state.canonical_id(id)).transpose()?;
    }
    state.tensors.insert(new_id, meta);
    Ok(())
}

fn set_i32_params(state: &mut BackendState, tid: u32, params: &[(usize, i32)]) -> Result<()> {
    let meta = state.tensor_mut(tid)?;
    for &(slot, val) in params {
        meta.set_op_param_i32(slot, val);
    }
    Ok(())
}

fn set_f32_params(state: &mut BackendState, tid: u32, params: &[(usize, f32)]) -> Result<()> {
    let meta = state.tensor_mut(tid)?;
    for &(slot, val) in params {
        meta.set_op_param_f32(slot, val);
    }
    Ok(())
}

fn is_scalar_query(func: &str) -> bool {
    matches!(
        func,
        "ggml_blck_size"
            | "ggml_type_size"
            | "ggml_type_name"
            | "ggml_op_name"
            | "ggml_unary_op_name"
            | "ggml_is_quantized"
            | "ggml_element_size"
            | "ggml_nbytes"
            | "ggml_nelements"
            | "ggml_nrows"
            | "ggml_is_matrix"
            | "ggml_is_vector"
            | "ggml_are_same_shape"
            | "ggml_get_name"
            | "ggml_op_desc"
            | "ggml_tensor_overhead"
            | "ggml_graph_overhead"
            | "ggml_graph_overhead_custom"
            | "ggml_graph_n_nodes"
            | "ggml_graph_size"
    )
}

fn dispatch_query(state: &mut BackendState, func: &str, args: &[Value], res: &Value) -> Result<()> {
    let val = match func {
        "ggml_blck_size" => ScalarValue::I64(blck_size(arg_i32(&args[0])?)),
        "ggml_type_size" => ScalarValue::U64(type_size(arg_i32(&args[0])?)),
        "ggml_type_name" => ScalarValue::String(type_name(arg_i32(&args[0])?).to_owned()),
        "ggml_op_name" => ScalarValue::String(op_name(arg_i32(&args[0])?).to_owned()),
        "ggml_unary_op_name" => ScalarValue::String(unary_op_name(arg_i32(&args[0])?).to_owned()),
        "ggml_is_quantized" => ScalarValue::Bool(is_quantized(arg_i32(&args[0])?)),
        "ggml_element_size" => ScalarValue::U64(type_size(t_meta(state, args, 0)?.kind)),
        "ggml_nbytes" => ScalarValue::U64(t_meta(state, args, 0)?.bytes),
        "ggml_nelements" => ScalarValue::I64(t_meta(state, args, 0)?.nelements()),
        "ggml_nrows" => ScalarValue::I64(t_meta(state, args, 0)?.nrows()),
        "ggml_is_matrix" => {
            let dims = t_meta(state, args, 0)?.ne;
            ScalarValue::Bool(dims[2] == 1 && dims[3] == 1)
        }
        "ggml_is_vector" => {
            let dims = t_meta(state, args, 0)?.ne;
            ScalarValue::Bool(dims[1] == 1 && dims[2] == 1 && dims[3] == 1)
        }
        "ggml_are_same_shape" => {
            ScalarValue::Bool(t_meta(state, args, 0)?.ne == t_meta(state, args, 1)?.ne)
        }
        "ggml_get_name" => ScalarValue::String(t_meta(state, args, 0)?.name),
        "ggml_op_desc" => {
            let t = t_meta(state, args, 0)?;
            let desc = if t.op == OP_GLU {
                glu_op_name(t.op_param_i32(0)).to_owned()
            } else {
                t.op_description()
            };
            ScalarValue::String(desc)
        }
        "ggml_tensor_overhead" => ScalarValue::U64(400),
        "ggml_graph_overhead" => ScalarValue::U64(65536),
        "ggml_graph_overhead_custom" => {
            let size = arg_u64(&args[0])?;
            let grads = arg_bool(&args[1])?;
            ScalarValue::U64(size * 16 + if grads { 32768 } else { 0 } + 256)
        }
        "ggml_graph_n_nodes" => {
            let gid = arg_handle(&args[0], "Graph")?;
            let len = state
                .graphs
                .get(&gid)
                .ok_or_else(|| invalid("unknown graph"))?
                .nodes
                .len();
            ScalarValue::I32(i32::try_from(len)?)
        }
        "ggml_graph_size" => {
            let gid = arg_handle(&args[0], "Graph")?;
            let graph = state
                .graphs
                .get(&gid)
                .ok_or_else(|| invalid("unknown graph"))?;
            ScalarValue::I32(i32::try_from(graph.custom_size.max(graph.nodes.len()))?)
        }
        _ => return Err(invalid(format!("unknown scalar query {func}"))),
    };
    store_scalar(state, res, val)
}

fn expand_forward_dfs(
    state: &BackendState,
    tid: u32,
    visited: &mut BTreeSet<u32>,
    nodes: &mut Vec<u32>,
    leafs: &mut Vec<u32>,
) -> Result<()> {
    let cid = state.canonical_id(tid)?;
    if !visited.insert(cid) {
        return Ok(());
    }
    let meta = state.tensor(cid)?;
    if let Some(view_src) = meta.view_source
        && state.tensor(view_src)?.op != OP_NONE
    {
        expand_forward_dfs(state, view_src, visited, nodes, leafs)?;
    }
    for src in meta.sources.into_iter().flatten() {
        expand_forward_dfs(state, src, visited, nodes, leafs)?;
    }
    if meta.op == OP_NONE {
        leafs.push(cid);
    } else {
        nodes.push(cid);
    }
    Ok(())
}

fn build_forward_expand(state: &mut BackendState, gid: u32, tid: u32) -> Result<()> {
    let existing = state
        .graphs
        .get(&gid)
        .ok_or_else(|| invalid("unknown graph"))?;
    let mut visited: BTreeSet<u32> = existing
        .nodes
        .iter()
        .chain(existing.leafs.iter())
        .copied()
        .collect();
    let (mut new_nodes, mut new_leafs) = (Vec::new(), Vec::new());
    expand_forward_dfs(state, tid, &mut visited, &mut new_nodes, &mut new_leafs)?;
    let graph = state
        .graphs
        .get_mut(&gid)
        .ok_or_else(|| invalid("unknown graph"))?;
    graph.nodes.extend(new_nodes);
    graph.leafs.extend(new_leafs);
    Ok(())
}

fn build_backward_expand(
    state: &mut BackendState,
    ctx_id: u32,
    gid: u32,
    grad_accs: Option<&Value>,
) -> Result<()> {
    let (nodes, leafs) = {
        let graph = state
            .graphs
            .get(&gid)
            .ok_or_else(|| invalid("unknown graph"))?;
        (graph.nodes.clone(), graph.leafs.clone())
    };
    let supplied_accs = grad_accs
        .filter(|val| !val.is_null())
        .map(arg_tensor_array)
        .transpose()?
        .unwrap_or_default();
    let mut all_tensors = leafs;
    all_tensors.extend(&nodes);
    let mut grads = Vec::new();
    let mut acc_cursor = 0_usize;
    for &tid in &nodes {
        let meta = state.tensor(tid)?.clone();
        if (meta.flags & FLAG_PARAM) != 0 {
            let gid_t = if let Some(Some(user_acc)) = supplied_accs.get(acc_cursor) {
                state.canonical_id(*user_acc)?
            } else {
                let fresh = state.next_internal_id;
                state.next_internal_id += 1;
                alloc_tensor(state, ctx_id, fresh, TYPE_F32, meta.ne)?;
                fresh
            };
            acc_cursor += 1;
            grads.push((tid, gid_t));
        }
    }
    for &tid in &all_tensors {
        let meta = state.tensor(tid)?.clone();
        if (meta.flags & FLAG_PARAM) != 0 && !grads.iter().any(|&(k, _)| k == tid) {
            let fresh = state.next_internal_id;
            state.next_internal_id += 1;
            alloc_tensor(state, ctx_id, fresh, TYPE_F32, meta.ne)?;
            grads.push((tid, fresh));
        }
    }
    state
        .graphs
        .get_mut(&gid)
        .ok_or_else(|| invalid("unknown graph"))?
        .gradients = grads;
    Ok(())
}

fn dispatch_graph_ops(
    state: &mut BackendState,
    func: &str,
    args: &[Value],
    res: &Value,
) -> Result<()> {
    match func {
        "ggml_new_graph" | "ggml_new_graph_custom" => {
            let _ctx_id = arg_handle(&args[0], "Context")?;
            let custom_size = if func == "ggml_new_graph" {
                2048
            } else {
                usize::try_from(arg_u64(&args[1])?)?
            };
            if func == "ggml_new_graph_custom" {
                let _has_grads = arg_bool(&args[2])?;
            }
            let record = GraphRecord {
                nodes: Vec::new(),
                leafs: Vec::new(),
                gradients: Vec::new(),
                custom_size,
            };
            state.graphs.insert(res_graph(res)?, record);
        }
        "ggml_build_forward_expand" => {
            build_forward_expand(state, arg_handle(&args[0], "Graph")?, t_id(args, 1)?)?;
        }
        "ggml_build_backward_expand" => build_backward_expand(
            state,
            arg_handle(&args[0], "Context")?,
            arg_handle(&args[1], "Graph")?,
            args.get(2),
        )?,
        "ggml_graph_add_node" => {
            let cid = state.canonical_id(t_id(args, 1)?)?;
            let graph = state
                .graphs
                .get_mut(&arg_handle(&args[0], "Graph")?)
                .ok_or_else(|| invalid("unknown graph"))?;
            if !graph.nodes.contains(&cid) {
                graph.nodes.push(cid);
            }
        }
        "ggml_graph_cpy" => {
            let cloned = {
                let src = state
                    .graphs
                    .get(&arg_handle(&args[0], "Graph")?)
                    .ok_or_else(|| invalid("unknown src graph"))?;
                (src.nodes.clone(), src.leafs.clone(), src.gradients.clone())
            };
            let dst = state
                .graphs
                .get_mut(&arg_handle(&args[1], "Graph")?)
                .ok_or_else(|| invalid("unknown dst graph"))?;
            (dst.nodes, dst.leafs, dst.gradients) = cloned;
        }
        "ggml_graph_reset" => {
            let grads = state
                .graphs
                .get(&arg_handle(&args[0], "Graph")?)
                .ok_or_else(|| invalid("unknown graph"))?
                .gradients
                .clone();
            for &(_, grad_tid) in &grads {
                set_tensor_zero(state, grad_tid)?;
            }
        }
        _ => {
            eval_ops::eval_graph(
                state,
                arg_handle(&args[0], "Backend")?,
                arg_handle(&args[1], "Graph")?,
            )?;
            store_scalar(state, res, ScalarValue::I32(STATUS_SUCCESS))?;
        }
    }
    Ok(())
}

fn dispatch_meta_ops(
    state: &mut BackendState,
    func: &str,
    args: &[Value],
    res: &Value,
) -> Result<()> {
    match func {
        "ggml_backend_buffer_set_usage" => {
            state
                .buffer_usage
                .insert(arg_handle(&args[0], "Buffer")?, arg_i32(&args[1])?);
        }
        "ggml_set_name" => {
            let tid = t_id(args, 0)?;
            state.tensor_mut(tid)?.name = arg_str(&args[1])?.to_owned();
            if let Ok(out_id) = res_tensor(res) {
                state.register_alias(tid, out_id)?;
            }
        }
        "ggml_get_tensor" => {
            let ctx_id = arg_handle(&args[0], "Context")?;
            let name = arg_str(&args[1])?;
            let found = state
                .contexts
                .get(&ctx_id)
                .ok_or_else(|| invalid("unknown context"))?
                .tensors
                .iter()
                .copied()
                .find(|&id| state.tensors.get(&id).is_some_and(|meta| meta.name == name))
                .ok_or_else(|| invalid(format!("ggml_get_tensor cannot find {name}")))?;
            state.register_alias(found, res_tensor(res)?)?;
        }
        "ggml_set_param" => state.tensor_mut(t_id(args, 0)?)?.flags |= FLAG_PARAM,
        "ggml_set_loss" => state.tensor_mut(t_id(args, 0)?)?.flags |= FLAG_LOSS,
        "ggml_mul_mat_set_hint" => {
            set_i32_params(state, t_id(args, 0)?, &[(0, arg_i32(&args[1])?)])?;
        }
        "ggml_flash_attn_ext_add_sinks" => {
            let sinks = arg_opt_tensor(&args[1])?
                .map(|id| state.canonical_id(id))
                .transpose()?;
            state.tensor_mut(t_id(args, 0)?)?.sources[4] = sinks;
        }
        "ggml_flash_attn_ext_set_prec" => {
            set_i32_params(state, t_id(args, 0)?, &[(3, arg_i32(&args[1])?)])?;
        }
        "ggml_soft_max_add_sinks" => {
            let sinks = arg_opt_tensor(&args[1])?
                .map(|id| state.canonical_id(id))
                .transpose()?;
            state.tensor_mut(t_id(args, 0)?)?.sources[2] = sinks;
        }
        "ggml_gated_delta_net_set_raw_gates" => {
            let (dt_bias, alpha) = (
                state.canonical_id(t_id(args, 1)?)?,
                state.canonical_id(t_id(args, 2)?)?,
            );
            let meta = state.tensor_mut(t_id(args, 0)?)?;
            (meta.sources[6], meta.sources[7]) = (Some(dt_bias), Some(alpha));
        }
        _ => {
            let tid = t_id(args, 0)?;
            set_i32_params(state, tid, &[(15, arg_i32(&args[1])?)])?;
            if let Ok(out_id) = res_tensor(res) {
                state.register_alias(tid, out_id)?;
            }
        }
    }
    Ok(())
}

fn dispatch_new_tensors(
    state: &mut BackendState,
    func: &str,
    args: &[Value],
    res: &Value,
) -> Result<()> {
    let (ctx_id, out_id, kind) = (
        arg_handle(&args[0], "Context")?,
        res_tensor(res)?,
        arg_i32(&args[1])?,
    );
    let dims = match func {
        "ggml_new_tensor" => {
            let n_dims = usize::try_from(arg_i32(&args[2])?)?;
            let arr = arg_i64_array(&args[3])?;
            let mut dims = [1_i64; GGML_MAX_DIMS];
            let count = n_dims.min(GGML_MAX_DIMS);
            dims[..count].copy_from_slice(&arr[..count]);
            dims
        }
        "ggml_new_tensor_1d" => [arg_i64(&args[2])?, 1, 1, 1],
        "ggml_new_tensor_2d" => [arg_i64(&args[2])?, arg_i64(&args[3])?, 1, 1],
        "ggml_new_tensor_3d" => [
            arg_i64(&args[2])?,
            arg_i64(&args[3])?,
            arg_i64(&args[4])?,
            1,
        ],
        _ => [
            arg_i64(&args[2])?,
            arg_i64(&args[3])?,
            arg_i64(&args[4])?,
            arg_i64(&args[5])?,
        ],
    };
    alloc_tensor(state, ctx_id, out_id, kind, dims)?;
    Ok(())
}

fn apply_view_fields(
    state: &mut BackendState,
    ctx_id: u32,
    out_id: u32,
    src: &TensorMeta,
    op_code: i32,
    geom: ([i64; 4], [u64; 4]),
    rel_off: u64,
) -> Result<()> {
    let (dims, strides) = geom;
    let mut meta = alloc_tensor(state, ctx_id, out_id, src.kind, dims)?;
    meta.nb = strides;
    meta.op = op_code;
    meta.view_source = Some(src.view_source.unwrap_or(src.id));
    meta.view_offset = src.view_offset + rel_off;
    meta.sources[0] = Some(src.id);
    meta.refresh_bytes()?;
    state.tensors.insert(out_id, meta);
    Ok(())
}

fn dispatch_view_1_2(
    state: &mut BackendState,
    func: &str,
    args: &[Value],
    res: &Value,
) -> Result<()> {
    let (ctx_id, out_id, src) = (
        arg_handle(&args[0], "Context")?,
        res_tensor(res)?,
        t_meta(state, args, 1)?,
    );
    match func {
        "ggml_view_tensor" => {
            apply_view_fields(state, ctx_id, out_id, &src, OP_VIEW, (src.ne, src.nb), 0)
        }
        "ggml_view_1d" => apply_view_fields(
            state,
            ctx_id,
            out_id,
            &src,
            OP_VIEW,
            ([arg_i64(&args[2])?, 1, 1, 1], src.nb),
            arg_u64(&args[3])?,
        ),
        "ggml_view_2d" => {
            let (ne1, nb1) = (arg_i64(&args[3])?, arg_u64(&args[4])?);
            let strides = [src.nb[0], nb1, nb1 * u64::try_from(ne1)?, src.nb[3]];
            apply_view_fields(
                state,
                ctx_id,
                out_id,
                &src,
                OP_VIEW,
                ([arg_i64(&args[2])?, ne1, 1, 1], strides),
                arg_u64(&args[5])?,
            )
        }
        "ggml_view_3d" => {
            let (ne2, nb1, nb2) = (arg_i64(&args[4])?, arg_u64(&args[5])?, arg_u64(&args[6])?);
            let strides = [src.nb[0], nb1, nb2, nb2 * u64::try_from(ne2)?];
            apply_view_fields(
                state,
                ctx_id,
                out_id,
                &src,
                OP_VIEW,
                ([arg_i64(&args[2])?, arg_i64(&args[3])?, ne2, 1], strides),
                arg_u64(&args[7])?,
            )
        }
        _ => {
            let dims = [
                arg_i64(&args[2])?,
                arg_i64(&args[3])?,
                arg_i64(&args[4])?,
                arg_i64(&args[5])?,
            ];
            let strides = [
                src.nb[0],
                arg_u64(&args[6])?,
                arg_u64(&args[7])?,
                arg_u64(&args[8])?,
            ];
            apply_view_fields(
                state,
                ctx_id,
                out_id,
                &src,
                OP_VIEW,
                (dims, strides),
                arg_u64(&args[9])?,
            )
        }
    }
}

fn dispatch_reshape_transpose_permute(
    state: &mut BackendState,
    func: &str,
    args: &[Value],
    res: &Value,
) -> Result<()> {
    let (ctx_id, out_id, src) = (
        arg_handle(&args[0], "Context")?,
        res_tensor(res)?,
        t_meta(state, args, 1)?,
    );
    match func {
        "ggml_reshape_2d" | "ggml_reshape_3d" | "ggml_reshape_4d" => {
            let dims = match func {
                "ggml_reshape_2d" => [arg_i64(&args[2])?, arg_i64(&args[3])?, 1, 1],
                "ggml_reshape_3d" => [
                    arg_i64(&args[2])?,
                    arg_i64(&args[3])?,
                    arg_i64(&args[4])?,
                    1,
                ],
                _ => [
                    arg_i64(&args[2])?,
                    arg_i64(&args[3])?,
                    arg_i64(&args[4])?,
                    arg_i64(&args[5])?,
                ],
            };
            let mut meta = alloc_tensor(state, ctx_id, out_id, src.kind, dims)?;
            (meta.op, meta.view_source, meta.view_offset, meta.sources[0]) = (
                OP_RESHAPE,
                Some(src.view_source.unwrap_or(src.id)),
                src.view_offset,
                Some(src.id),
            );
            meta.refresh_bytes()?;
            state.tensors.insert(out_id, meta);
            Ok(())
        }
        "ggml_transpose" => {
            let dims = [src.ne[1], src.ne[0], src.ne[2], src.ne[3]];
            let strides = [src.nb[1], src.nb[0], src.nb[2], src.nb[3]];
            apply_view_fields(
                state,
                ctx_id,
                out_id,
                &src,
                OP_TRANSPOSE,
                (dims, strides),
                0,
            )
        }
        _ => {
            let axes = [
                usize::try_from(arg_i32(&args[2])?)?,
                usize::try_from(arg_i32(&args[3])?)?,
                usize::try_from(arg_i32(&args[4])?)?,
                usize::try_from(arg_i32(&args[5])?)?,
            ];
            let (mut dims, mut strides) = ([1_i64; 4], [0_u64; 4]);
            for (src_ax, &dst_ax) in axes.iter().enumerate() {
                dims[dst_ax] = src.ne[src_ax];
                strides[dst_ax] = src.nb[src_ax];
            }
            apply_view_fields(state, ctx_id, out_id, &src, OP_PERMUTE, (dims, strides), 0)?;
            for (slot, &ax) in axes.iter().enumerate() {
                set_i32_params(state, out_id, &[(slot, i32::try_from(ax)?)])?;
            }
            Ok(())
        }
    }
}

fn unary_code_for(func: &str) -> Option<i32> {
    Some(match func {
        "ggml_relu" => UNARY_OP_RELU,
        "ggml_sigmoid" => UNARY_OP_SIGMOID,
        "ggml_silu" => UNARY_OP_SILU,
        "ggml_tanh" => UNARY_OP_TANH,
        "ggml_floor" => UNARY_OP_FLOOR,
        "ggml_ceil" => UNARY_OP_CEIL,
        "ggml_round" => UNARY_OP_ROUND,
        "ggml_trunc" => UNARY_OP_TRUNC,
        "ggml_softplus" => UNARY_OP_SOFTPLUS,
        _ => return None,
    })
}

fn math_unary_op_for(func: &str) -> Option<i32> {
    Some(match func {
        "ggml_sqr" => OP_SQR,
        "ggml_sqrt" => OP_SQRT,
        "ggml_log" => OP_LOG,
        "ggml_sin" => OP_SIN,
        "ggml_cos" => OP_COS,
        _ => return None,
    })
}

fn dispatch_binary_arith(
    state: &mut BackendState,
    func: &str,
    args: &[Value],
    res: &Value,
) -> Result<()> {
    let (ctx_id, out_id) = (arg_handle(&args[0], "Context")?, res_tensor(res)?);
    let (lhs_id, rhs_id) = (t_id(args, 1)?, t_id(args, 2)?);
    let lhs = state.tensor(lhs_id)?.clone();
    match func {
        "ggml_add" | "ggml_sub" | "ggml_mul" | "ggml_div" => {
            let op_code = match func {
                "ggml_add" => OP_ADD,
                "ggml_sub" => OP_SUB,
                "ggml_mul" => OP_MUL,
                _ => OP_DIV,
            };
            finish_op(
                state,
                ctx_id,
                out_id,
                lhs.kind,
                lhs.ne,
                op_code,
                &[Some(lhs_id), Some(rhs_id)],
            )
        }
        "ggml_add_inplace" | "ggml_sub_inplace" | "ggml_mul_inplace" | "ggml_div_inplace" => {
            let op_code = match func {
                "ggml_add_inplace" => OP_ADD,
                "ggml_sub_inplace" => OP_SUB,
                "ggml_mul_inplace" => OP_MUL,
                _ => OP_DIV,
            };
            finish_inplace(
                state,
                ctx_id,
                out_id,
                lhs_id,
                op_code,
                &[Some(lhs_id), Some(rhs_id)],
            )
        }
        "ggml_add_id" => finish_op(
            state,
            ctx_id,
            out_id,
            lhs.kind,
            lhs.ne,
            OP_ADD_ID,
            &[Some(lhs_id), Some(rhs_id), Some(t_id(args, 3)?)],
        ),
        _ => {
            let (nb1, nb2, nb3, off) = (
                i32::try_from(arg_u64(&args[3])?)?,
                i32::try_from(arg_u64(&args[4])?)?,
                i32::try_from(arg_u64(&args[5])?)?,
                i32::try_from(arg_u64(&args[6])?)?,
            );
            let (op_code, inplace) = (
                if func == "ggml_acc" { OP_ACC } else { OP_SET },
                i32::from(func == "ggml_set_inplace"),
            );
            if inplace == 1 {
                finish_inplace(
                    state,
                    ctx_id,
                    out_id,
                    lhs_id,
                    op_code,
                    &[Some(lhs_id), Some(rhs_id)],
                )?;
            } else {
                finish_op(
                    state,
                    ctx_id,
                    out_id,
                    lhs.kind,
                    lhs.ne,
                    op_code,
                    &[Some(lhs_id), Some(rhs_id)],
                )?;
            }
            set_i32_params(
                state,
                out_id,
                &[(0, nb1), (1, nb2), (2, nb3), (3, off), (4, inplace)],
            )
        }
    }
}

fn dispatch_copy_repeat_concat(
    state: &mut BackendState,
    func: &str,
    args: &[Value],
    res: &Value,
) -> Result<()> {
    let (ctx_id, out_id) = (arg_handle(&args[0], "Context")?, res_tensor(res)?);
    let src = t_meta(state, args, 1)?;
    match func {
        "ggml_cpy" => finish_inplace(
            state,
            ctx_id,
            out_id,
            t_id(args, 2)?,
            OP_CPY,
            &[Some(src.id), Some(t_id(args, 2)?)],
        ),
        "ggml_dup" | "ggml_cont" => finish_op(
            state,
            ctx_id,
            out_id,
            src.kind,
            src.ne,
            if func == "ggml_dup" { OP_DUP } else { OP_CONT },
            &[Some(src.id)],
        ),
        "ggml_repeat" | "ggml_repeat_back" => finish_op(
            state,
            ctx_id,
            out_id,
            src.kind,
            t_meta(state, args, 2)?.ne,
            if func == "ggml_repeat" {
                OP_REPEAT
            } else {
                OP_REPEAT_BACK
            },
            &[Some(src.id)],
        ),
        "ggml_repeat_4d" => finish_op(
            state,
            ctx_id,
            out_id,
            src.kind,
            [
                arg_i64(&args[2])?,
                arg_i64(&args[3])?,
                arg_i64(&args[4])?,
                arg_i64(&args[5])?,
            ],
            OP_REPEAT,
            &[Some(src.id)],
        ),
        "ggml_concat" => {
            let (rhs, dim) = (
                t_meta(state, args, 2)?,
                usize::try_from(arg_i32(&args[3])?)?,
            );
            let mut dims = src.ne;
            dims[dim] += rhs.ne[dim];
            finish_op(
                state,
                ctx_id,
                out_id,
                src.kind,
                dims,
                OP_CONCAT,
                &[Some(src.id), Some(rhs.id)],
            )?;
            set_i32_params(state, out_id, &[(0, i32::try_from(dim)?)])
        }
        _ => finish_op(
            state,
            ctx_id,
            out_id,
            TYPE_I64,
            [1, 1, 1, 1],
            OP_COUNT_EQUAL,
            &[Some(src.id), Some(t_id(args, 2)?)],
        ),
    }
}

fn dispatch_unary_and_scalar_ops(
    state: &mut BackendState,
    func: &str,
    args: &[Value],
    res: &Value,
) -> Result<()> {
    let (ctx_id, out_id, src) = (
        arg_handle(&args[0], "Context")?,
        res_tensor(res)?,
        state
            .tensor(
                args.get(1)
                    .and_then(|v| v.get("Tensor"))
                    .and_then(Value::as_u64)
                    .and_then(|v| u32::try_from(v).ok())
                    .unwrap_or(0),
            )
            .ok()
            .cloned(),
    );
    if let Some(uop) = unary_code_for(func) {
        let s = src.ok_or_else(|| invalid("unary missing src"))?;
        finish_op(state, ctx_id, out_id, s.kind, s.ne, OP_UNARY, &[Some(s.id)])?;
        return set_i32_params(state, out_id, &[(0, uop)]);
    }
    if let Some(op_code) = math_unary_op_for(func) {
        let s = src.ok_or_else(|| invalid("math unary missing src"))?;
        return finish_op(state, ctx_id, out_id, s.kind, s.ne, op_code, &[Some(s.id)]);
    }
    match func {
        "ggml_unary" => {
            let s = src.ok_or_else(|| invalid("unary missing src"))?;
            finish_op(state, ctx_id, out_id, s.kind, s.ne, OP_UNARY, &[Some(s.id)])?;
            set_i32_params(state, out_id, &[(0, arg_i32(&args[2])?)])
        }
        "ggml_silu_back" => {
            let grad = t_meta(state, args, 1)?;
            finish_op(
                state,
                ctx_id,
                out_id,
                grad.kind,
                grad.ne,
                OP_SILU_BACK,
                &[Some(grad.id), Some(t_id(args, 2)?)],
            )
        }
        "ggml_leaky_relu" => {
            let s = t_meta(state, args, 1)?;
            let (slope, inplace) = (arg_f32(&args[2])?, arg_bool(&args[3])?);
            if inplace {
                finish_inplace(state, ctx_id, out_id, s.id, OP_LEAKY_RELU, &[Some(s.id)])?;
            } else {
                finish_op(
                    state,
                    ctx_id,
                    out_id,
                    s.kind,
                    s.ne,
                    OP_LEAKY_RELU,
                    &[Some(s.id)],
                )?;
            }
            set_f32_params(state, out_id, &[(0, slope)])
        }
        _ => dispatch_scale_clamp_arange(state, ctx_id, out_id, func, args),
    }
}

fn dispatch_scale_clamp_arange(
    state: &mut BackendState,
    ctx_id: u32,
    out_id: u32,
    func: &str,
    args: &[Value],
) -> Result<()> {
    match func {
        "ggml_xielu" => {
            let s = t_meta(state, args, 1)?;
            finish_op(state, ctx_id, out_id, s.kind, s.ne, OP_UNARY, &[Some(s.id)])?;
            set_i32_params(state, out_id, &[(0, UNARY_OP_XIELU)])?;
            set_f32_params(
                state,
                out_id,
                &[
                    (1, arg_f32(&args[2])?),
                    (2, arg_f32(&args[3])?),
                    (3, arg_f32(&args[4])?),
                    (4, arg_f32(&args[5])?),
                ],
            )
        }
        "ggml_clamp" => {
            let s = t_meta(state, args, 1)?;
            finish_op(state, ctx_id, out_id, s.kind, s.ne, OP_CLAMP, &[Some(s.id)])?;
            set_f32_params(
                state,
                out_id,
                &[(0, arg_f32(&args[2])?), (1, arg_f32(&args[3])?)],
            )
        }
        "ggml_scale" | "ggml_scale_bias" | "ggml_scale_bias_inplace" => {
            let s = t_meta(state, args, 1)?;
            let (scale_v, bias_v) = (
                arg_f32(&args[2])?,
                if func == "ggml_scale" {
                    0.0
                } else {
                    arg_f32(&args[3])?
                },
            );
            if func == "ggml_scale_bias_inplace" {
                finish_inplace(state, ctx_id, out_id, s.id, OP_SCALE, &[Some(s.id)])?;
            } else {
                finish_op(state, ctx_id, out_id, s.kind, s.ne, OP_SCALE, &[Some(s.id)])?;
            }
            set_f32_params(state, out_id, &[(0, scale_v), (1, bias_v)])
        }
        "ggml_fill" => {
            let s = t_meta(state, args, 1)?;
            finish_op(state, ctx_id, out_id, s.kind, s.ne, OP_FILL, &[Some(s.id)])?;
            set_f32_params(state, out_id, &[(0, arg_f32(&args[2])?)])
        }
        "ggml_arange" => {
            let (start, limit_val, delta) =
                (arg_f32(&args[1])?, arg_f32(&args[2])?, arg_f32(&args[3])?);
            let n_elem = if delta == 0.0 {
                0.0
            } else {
                ((f64::from(limit_val) - f64::from(start)) / f64::from(delta))
                    .ceil()
                    .max(0.0)
            };
            finish_op(
                state,
                ctx_id,
                out_id,
                TYPE_F32,
                [f32_to_i64_trunc(f64_to_f32(n_elem)), 1, 1, 1],
                OP_ARANGE,
                &[],
            )?;
            set_f32_params(state, out_id, &[(0, start), (1, limit_val), (2, delta)])
        }
        _ => dispatch_glu(state, ctx_id, out_id, func, args),
    }
}

fn dispatch_glu(
    state: &mut BackendState,
    ctx_id: u32,
    out_id: u32,
    func: &str,
    args: &[Value],
) -> Result<()> {
    let s = t_meta(state, args, 1)?;
    match func {
        "ggml_glu" => {
            finish_op(
                state,
                ctx_id,
                out_id,
                s.kind,
                [s.ne[0] / 2, s.ne[1], s.ne[2], s.ne[3]],
                OP_GLU,
                &[Some(s.id)],
            )?;
            set_i32_params(
                state,
                out_id,
                &[(0, arg_i32(&args[2])?), (1, i32::from(arg_bool(&args[3])?))],
            )
        }
        "ggml_glu_split" | "ggml_swiglu_split" | "ggml_swiglu_oai" => {
            let gop = if func == "ggml_glu_split" {
                arg_i32(&args[3])?
            } else if func == "ggml_swiglu_split" {
                2
            } else {
                3
            };
            finish_op(
                state,
                ctx_id,
                out_id,
                s.kind,
                s.ne,
                OP_GLU,
                &[Some(s.id), Some(t_id(args, 2)?)],
            )?;
            set_i32_params(state, out_id, &[(0, gop), (1, 0)])?;
            if func == "ggml_swiglu_oai" {
                set_f32_params(
                    state,
                    out_id,
                    &[(2, arg_f32(&args[3])?), (3, arg_f32(&args[4])?)],
                )?;
            }
            Ok(())
        }
        _ => dispatch_reduce_diag_tri(state, ctx_id, out_id, func, &s, args),
    }
}

fn dispatch_reduce_diag_tri(
    state: &mut BackendState,
    ctx_id: u32,
    out_id: u32,
    func: &str,
    s: &TensorMeta,
    args: &[Value],
) -> Result<()> {
    match func {
        "ggml_sum" => finish_op(
            state,
            ctx_id,
            out_id,
            s.kind,
            [1, 1, 1, 1],
            OP_SUM,
            &[Some(s.id)],
        ),
        "ggml_sum_rows" | "ggml_mean" => finish_op(
            state,
            ctx_id,
            out_id,
            s.kind,
            [1, s.ne[1], s.ne[2], s.ne[3]],
            if func == "ggml_sum_rows" {
                OP_SUM_ROWS
            } else {
                OP_MEAN
            },
            &[Some(s.id)],
        ),
        "ggml_cumsum" => finish_op(
            state,
            ctx_id,
            out_id,
            s.kind,
            s.ne,
            OP_CUMSUM,
            &[Some(s.id)],
        ),
        "ggml_argmax" => finish_op(
            state,
            ctx_id,
            out_id,
            TYPE_I32,
            [1, s.ne[1], s.ne[2], s.ne[3]],
            OP_ARGMAX,
            &[Some(s.id)],
        ),
        "ggml_argsort" => {
            finish_op(
                state,
                ctx_id,
                out_id,
                TYPE_I32,
                s.ne,
                OP_ARGSORT,
                &[Some(s.id)],
            )?;
            set_i32_params(state, out_id, &[(0, arg_i32(&args[2])?)])
        }
        "ggml_argsort_top_k" | "ggml_top_k" => {
            finish_op(
                state,
                ctx_id,
                out_id,
                TYPE_I32,
                [i64::from(arg_i32(&args[2])?), s.ne[1], s.ne[2], s.ne[3]],
                if func == "ggml_top_k" {
                    OP_TOP_K
                } else {
                    OP_ARGSORT
                },
                &[Some(s.id)],
            )?;
            set_i32_params(state, out_id, &[(0, 1)])
        }
        "ggml_diag" => finish_op(
            state,
            ctx_id,
            out_id,
            s.kind,
            [s.ne[0], s.ne[0], s.ne[1], s.ne[2]],
            OP_DIAG,
            &[Some(s.id)],
        ),
        "ggml_diag_mask_inf" => {
            finish_op(
                state,
                ctx_id,
                out_id,
                s.kind,
                s.ne,
                OP_DIAG_MASK_INF,
                &[Some(s.id)],
            )?;
            set_i32_params(state, out_id, &[(0, arg_i32(&args[2])?)])
        }
        "ggml_tri" => {
            finish_op(state, ctx_id, out_id, s.kind, s.ne, OP_TRI, &[Some(s.id)])?;
            set_i32_params(state, out_id, &[(0, arg_i32(&args[2])?)])
        }
        _ => Err(invalid(format!("unknown unary/glu/reduce function {func}"))),
    }
}

fn dispatch_norm_ops(
    state: &mut BackendState,
    func: &str,
    args: &[Value],
    res: &Value,
) -> Result<()> {
    let (ctx_id, out_id, src) = (
        arg_handle(&args[0], "Context")?,
        res_tensor(res)?,
        t_meta(state, args, 1)?,
    );
    match func {
        "ggml_norm" | "ggml_rms_norm" | "ggml_rms_norm_inplace" | "ggml_l2_norm" => {
            let op_code = match func {
                "ggml_norm" => OP_NORM,
                "ggml_l2_norm" => OP_L2_NORM,
                _ => OP_RMS_NORM,
            };
            if func == "ggml_rms_norm_inplace" {
                finish_inplace(state, ctx_id, out_id, src.id, op_code, &[Some(src.id)])?;
            } else {
                finish_op(
                    state,
                    ctx_id,
                    out_id,
                    src.kind,
                    src.ne,
                    op_code,
                    &[Some(src.id)],
                )?;
            }
            set_f32_params(state, out_id, &[(0, arg_f32(&args[2])?)])
        }
        "ggml_rms_norm_back" => {
            finish_op(
                state,
                ctx_id,
                out_id,
                src.kind,
                src.ne,
                OP_RMS_NORM_BACK,
                &[Some(src.id), Some(t_id(args, 2)?)],
            )?;
            set_f32_params(state, out_id, &[(0, arg_f32(&args[3])?)])
        }
        _ => {
            finish_op(
                state,
                ctx_id,
                out_id,
                src.kind,
                src.ne,
                OP_GROUP_NORM,
                &[Some(src.id)],
            )?;
            set_i32_params(state, out_id, &[(0, arg_i32(&args[2])?)])?;
            set_f32_params(state, out_id, &[(1, arg_f32(&args[3])?)])
        }
    }
}

fn dispatch_mat_and_rows(
    state: &mut BackendState,
    func: &str,
    args: &[Value],
    res: &Value,
) -> Result<()> {
    let (ctx_id, out_id) = (arg_handle(&args[0], "Context")?, res_tensor(res)?);
    match func {
        "ggml_mul_mat" => {
            let (lhs, rhs) = (t_meta(state, args, 1)?, t_meta(state, args, 2)?);
            finish_op(
                state,
                ctx_id,
                out_id,
                TYPE_F32,
                [
                    lhs.ne[1],
                    rhs.ne[1],
                    lhs.ne[2].max(rhs.ne[2]),
                    lhs.ne[3].max(rhs.ne[3]),
                ],
                OP_MUL_MAT,
                &[Some(lhs.id), Some(rhs.id)],
            )
        }
        "ggml_mul_mat_id" => {
            let (mat, ids) = (t_meta(state, args, 1)?, t_meta(state, args, 3)?);
            finish_op(
                state,
                ctx_id,
                out_id,
                TYPE_F32,
                [mat.ne[1], ids.ne[0], ids.ne[1], 1],
                OP_MUL_MAT_ID,
                &[Some(mat.id), Some(t_id(args, 2)?), Some(ids.id)],
            )
        }
        "ggml_out_prod" => {
            let (lhs, rhs) = (t_meta(state, args, 1)?, t_meta(state, args, 2)?);
            finish_op(
                state,
                ctx_id,
                out_id,
                TYPE_F32,
                [lhs.ne[0], rhs.ne[0], lhs.ne[2], lhs.ne[3]],
                OP_OUT_PROD,
                &[Some(lhs.id), Some(rhs.id)],
            )
        }
        "ggml_get_rows" => {
            let (src, rows) = (t_meta(state, args, 1)?, t_meta(state, args, 2)?);
            finish_op(
                state,
                ctx_id,
                out_id,
                if src.kind == TYPE_I32 {
                    TYPE_I32
                } else {
                    TYPE_F32
                },
                [src.ne[0], rows.ne[0], rows.ne[1], rows.ne[2]],
                OP_GET_ROWS,
                &[Some(src.id), Some(rows.id)],
            )
        }
        "ggml_get_rows_back" => {
            let shape_ref = t_meta(state, args, 3)?;
            finish_op(
                state,
                ctx_id,
                out_id,
                TYPE_F32,
                shape_ref.ne,
                OP_GET_ROWS_BACK,
                &[
                    Some(t_id(args, 1)?),
                    Some(t_id(args, 2)?),
                    Some(shape_ref.id),
                ],
            )
        }
        _ => {
            let dst = t_meta(state, args, 1)?;
            finish_inplace(
                state,
                ctx_id,
                out_id,
                dst.id,
                OP_SET_ROWS,
                &[Some(t_id(args, 2)?), Some(t_id(args, 3)?), Some(dst.id)],
            )
        }
    }
}

fn dispatch_softmax_and_flash(
    state: &mut BackendState,
    func: &str,
    args: &[Value],
    res: &Value,
) -> Result<()> {
    let (ctx_id, out_id) = (arg_handle(&args[0], "Context")?, res_tensor(res)?);
    match func {
        "ggml_soft_max" | "ggml_soft_max_ext" | "ggml_soft_max_ext_inplace" => {
            let src = t_meta(state, args, 1)?;
            let mask = args.get(2).map(arg_opt_tensor).transpose()?.flatten();
            let (scale, max_bias) = (
                if func == "ggml_soft_max" {
                    1.0
                } else {
                    arg_f32(&args[3])?
                },
                if func == "ggml_soft_max" {
                    0.0
                } else {
                    arg_f32(&args[4])?
                },
            );
            if func == "ggml_soft_max_ext_inplace" {
                finish_inplace(
                    state,
                    ctx_id,
                    out_id,
                    src.id,
                    OP_SOFT_MAX,
                    &[Some(src.id), mask],
                )?;
            } else {
                finish_op(
                    state,
                    ctx_id,
                    out_id,
                    src.kind,
                    src.ne,
                    OP_SOFT_MAX,
                    &[Some(src.id), mask],
                )?;
            }
            set_f32_params(state, out_id, &[(0, scale), (1, max_bias)])
        }
        "ggml_soft_max_ext_back" => {
            let lhs = t_meta(state, args, 1)?;
            finish_op(
                state,
                ctx_id,
                out_id,
                lhs.kind,
                lhs.ne,
                OP_SOFT_MAX_BACK,
                &[Some(lhs.id), Some(t_id(args, 2)?)],
            )?;
            set_f32_params(
                state,
                out_id,
                &[(0, arg_f32(&args[3])?), (1, arg_f32(&args[4])?)],
            )
        }
        _ => {
            let (q, v) = (t_meta(state, args, 1)?, t_meta(state, args, 3)?);
            finish_op(
                state,
                ctx_id,
                out_id,
                TYPE_F32,
                [v.ne[0], q.ne[2], q.ne[1], q.ne[3]],
                OP_FLASH_ATTN_EXT,
                &[
                    Some(q.id),
                    Some(t_id(args, 2)?),
                    Some(v.id),
                    arg_opt_tensor(&args[4])?,
                ],
            )?;
            set_f32_params(
                state,
                out_id,
                &[
                    (0, arg_f32(&args[5])?),
                    (1, arg_f32(&args[6])?),
                    (2, arg_f32(&args[7])?),
                ],
            )
        }
    }
}

fn dispatch_rope(state: &mut BackendState, func: &str, args: &[Value], res: &Value) -> Result<()> {
    let (ctx_id, out_id, src, pos) = (
        arg_handle(&args[0], "Context")?,
        res_tensor(res)?,
        t_meta(state, args, 1)?,
        t_id(args, 2)?,
    );
    let (is_ext, is_multi) = (
        func.contains("_ext") || func.contains("_multi"),
        func.contains("_multi"),
    );
    let op_code = if func.ends_with("_back") {
        OP_ROPE_BACK
    } else {
        OP_ROPE
    };
    let freq = if is_ext {
        arg_opt_tensor(&args[3])?
    } else {
        None
    };
    let idx = if is_ext { 4 } else { 3 };
    let n_dims = arg_i32(&args[idx])?;
    if func.ends_with("_inplace") {
        finish_inplace(
            state,
            ctx_id,
            out_id,
            src.id,
            op_code,
            &[Some(src.id), Some(pos), freq],
        )?;
    } else {
        finish_op(
            state,
            ctx_id,
            out_id,
            src.kind,
            src.ne,
            op_code,
            &[Some(src.id), Some(pos), freq],
        )?;
    }
    if !is_ext {
        set_i32_params(state, out_id, &[(1, n_dims), (2, arg_i32(&args[idx + 1])?)])?;
        return set_f32_params(
            state,
            out_id,
            &[(5, 10000.0), (6, 1.0), (8, 1.0), (9, 32.0), (10, 1.0)],
        );
    }
    let sections = if is_multi {
        arg_i32_array(&args[idx + 1])?
    } else {
        Vec::new()
    };
    let b = if is_multi { idx + 2 } else { idx + 1 };
    set_i32_params(
        state,
        out_id,
        &[
            (1, n_dims),
            (2, arg_i32(&args[b])?),
            (3, 0),
            (4, arg_i32(&args[b + 1])?),
        ],
    )?;
    set_f32_params(
        state,
        out_id,
        &[
            (5, arg_f32(&args[b + 2])?),
            (6, arg_f32(&args[b + 3])?),
            (7, arg_f32(&args[b + 4])?),
            (8, arg_f32(&args[b + 5])?),
            (9, arg_f32(&args[b + 6])?),
            (10, arg_f32(&args[b + 7])?),
        ],
    )?;
    for (slot, &sec) in sections.iter().take(4).enumerate() {
        set_i32_params(state, out_id, &[(11 + slot, sec)])?;
    }
    Ok(())
}

fn dispatch_im2col(
    state: &mut BackendState,
    ctx_id: u32,
    out_id: u32,
    kern: &TensorMeta,
    inp: &TensorMeta,
    args: &[Value],
) -> Result<()> {
    let p = (3..9)
        .map(|i| arg_i32(&args[i]))
        .collect::<Result<Vec<_>>>()?;
    let (is_2d, dst_kind) = (arg_bool(&args[9])?, arg_i32(&args[10])?);
    let ow = (inp.ne[0] + 2 * i64::from(p[2]) - i64::from(p[4]) * (kern.ne[0] - 1) - 1)
        / i64::from(p[0])
        + 1;
    let oh = if is_2d {
        (inp.ne[1] + 2 * i64::from(p[3]) - i64::from(p[5]) * (kern.ne[1] - 1) - 1) / i64::from(p[1])
            + 1
    } else {
        inp.ne[1]
    };
    let dims = if is_2d {
        [kern.ne[0] * kern.ne[1] * kern.ne[2], ow, oh, inp.ne[3]]
    } else {
        [kern.ne[0] * kern.ne[1], ow, inp.ne[2], inp.ne[3]]
    };
    finish_op(
        state,
        ctx_id,
        out_id,
        dst_kind,
        dims,
        OP_IM2COL,
        &[Some(kern.id), Some(inp.id)],
    )?;
    set_i32_params(
        state,
        out_id,
        &[
            (0, p[0]),
            (1, p[1]),
            (2, p[2]),
            (3, p[3]),
            (4, p[4]),
            (5, p[5]),
            (6, i32::from(is_2d)),
        ],
    )
}

fn dispatch_conv_ops(
    state: &mut BackendState,
    func: &str,
    args: &[Value],
    res: &Value,
) -> Result<()> {
    let (ctx_id, out_id, kern, inp) = (
        arg_handle(&args[0], "Context")?,
        res_tensor(res)?,
        t_meta(state, args, 1)?,
        t_meta(state, args, 2)?,
    );
    match func {
        "ggml_im2col" => dispatch_im2col(state, ctx_id, out_id, &kern, &inp, args),
        "ggml_im2col_3d" => {
            let (ic, params) = (
                arg_i64(&args[3])?,
                (4..13)
                    .map(|i| arg_i32(&args[i]))
                    .collect::<Result<Vec<_>>>()?,
            );
            let ow = (inp.ne[0] + 2 * i64::from(params[3])
                - i64::from(params[6]) * (kern.ne[0] - 1)
                - 1)
                / i64::from(params[0])
                + 1;
            let oh = (inp.ne[1] + 2 * i64::from(params[4])
                - i64::from(params[7]) * (kern.ne[1] - 1)
                - 1)
                / i64::from(params[1])
                + 1;
            let od = (inp.ne[2] / ic + 2 * i64::from(params[5])
                - i64::from(params[8]) * (kern.ne[2] / ic - 1)
                - 1)
                / i64::from(params[2])
                + 1;
            finish_op(
                state,
                ctx_id,
                out_id,
                arg_i32(&args[13])?,
                [kern.ne[0] * kern.ne[1] * kern.ne[2], ow, oh, od * inp.ne[3]],
                OP_IM2COL_3D,
                &[Some(kern.id), Some(inp.id)],
            )?;
            set_i32_params(
                state,
                out_id,
                &params.into_iter().enumerate().collect::<Vec<_>>(),
            )
        }
        "ggml_conv_2d_direct" | "ggml_conv_2d_dw_direct" => {
            let params = (3..9)
                .map(|i| arg_i32(&args[i]))
                .collect::<Result<Vec<_>>>()?;
            let ow = (inp.ne[0] + 2 * i64::from(params[2])
                - i64::from(params[4]) * (kern.ne[0] - 1)
                - 1)
                / i64::from(params[0])
                + 1;
            let oh = (inp.ne[1] + 2 * i64::from(params[3])
                - i64::from(params[5]) * (kern.ne[1] - 1)
                - 1)
                / i64::from(params[1])
                + 1;
            let is_dw = func == "ggml_conv_2d_dw_direct";
            finish_op(
                state,
                ctx_id,
                out_id,
                TYPE_F32,
                [
                    ow,
                    oh,
                    if is_dw { inp.ne[2] } else { kern.ne[3] },
                    inp.ne[3],
                ],
                if is_dw { OP_CONV_2D_DW } else { OP_CONV_2D },
                &[Some(kern.id), Some(inp.id)],
            )?;
            set_i32_params(
                state,
                out_id,
                &params.into_iter().enumerate().collect::<Vec<_>>(),
            )
        }
        _ => dispatch_conv_3d_transpose(state, ctx_id, out_id, func, &kern, &inp, args),
    }
}

fn dispatch_conv_3d_transpose(
    state: &mut BackendState,
    ctx_id: u32,
    out_id: u32,
    func: &str,
    kern: &TensorMeta,
    inp: &TensorMeta,
    args: &[Value],
) -> Result<()> {
    match func {
        "ggml_conv_3d_direct" => {
            let p = (3..15)
                .map(|i| arg_i32(&args[i]))
                .collect::<Result<Vec<_>>>()?;
            let ow = (inp.ne[0] + 2 * i64::from(p[3]) - i64::from(p[6]) * (kern.ne[0] - 1) - 1)
                / i64::from(p[0])
                + 1;
            let oh = (inp.ne[1] + 2 * i64::from(p[4]) - i64::from(p[7]) * (kern.ne[1] - 1) - 1)
                / i64::from(p[1])
                + 1;
            let (ic, batch, oc) = (i64::from(p[9]), i64::from(p[10]), i64::from(p[11]));
            let od = (inp.ne[2] / ic + 2 * i64::from(p[5])
                - i64::from(p[8]) * (kern.ne[2] / ic - 1)
                - 1)
                / i64::from(p[2])
                + 1;
            finish_op(
                state,
                ctx_id,
                out_id,
                TYPE_F32,
                [ow, oh, od * oc, batch],
                OP_CONV_3D,
                &[Some(kern.id), Some(inp.id)],
            )?;
            set_i32_params(
                state,
                out_id,
                &p.into_iter().enumerate().collect::<Vec<_>>(),
            )
        }
        "ggml_conv_transpose_1d" => {
            let (s0, p0, d0) = (arg_i32(&args[3])?, arg_i32(&args[4])?, arg_i32(&args[5])?);
            let ow = (inp.ne[0] - 1) * i64::from(s0) - 2 * i64::from(p0)
                + i64::from(d0) * (kern.ne[0] - 1)
                + 1;
            finish_op(
                state,
                ctx_id,
                out_id,
                TYPE_F32,
                [ow, kern.ne[1], inp.ne[2], inp.ne[3]],
                OP_CONV_TRANSPOSE_1D,
                &[Some(kern.id), Some(inp.id)],
            )?;
            set_i32_params(state, out_id, &[(0, s0), (1, p0), (2, d0)])
        }
        _ => {
            let stride = arg_i32(&args[3])?;
            let (ow, oh) = (
                (inp.ne[0] - 1) * i64::from(stride) + kern.ne[0],
                (inp.ne[1] - 1) * i64::from(stride) + kern.ne[1],
            );
            finish_op(
                state,
                ctx_id,
                out_id,
                TYPE_F32,
                [ow, oh, kern.ne[2], inp.ne[3]],
                OP_CONV_TRANSPOSE_2D,
                &[Some(kern.id), Some(inp.id)],
            )?;
            set_i32_params(state, out_id, &[(0, stride)])
        }
    }
}

fn dispatch_pool_and_upscale(
    state: &mut BackendState,
    func: &str,
    args: &[Value],
    res: &Value,
) -> Result<()> {
    let (ctx_id, out_id, src) = (
        arg_handle(&args[0], "Context")?,
        res_tensor(res)?,
        t_meta(state, args, 1)?,
    );
    match func {
        "ggml_col2im_1d" => {
            let (s0, oc, p0) = (arg_i32(&args[2])?, arg_i64(&args[3])?, arg_i32(&args[4])?);
            finish_op(
                state,
                ctx_id,
                out_id,
                src.kind,
                [src.ne[1] * i64::from(s0), oc, src.ne[2], src.ne[3]],
                OP_COL2IM_1D,
                &[Some(src.id)],
            )?;
            set_i32_params(state, out_id, &[(0, s0), (1, i32::try_from(oc)?), (2, p0)])
        }
        "ggml_pool_1d" => {
            let (pop, k0, s0, p0) = (
                arg_i32(&args[2])?,
                arg_i32(&args[3])?,
                arg_i32(&args[4])?,
                arg_i32(&args[5])?,
            );
            let ow = (src.ne[0] + 2 * i64::from(p0) - i64::from(k0)) / i64::from(s0) + 1;
            finish_op(
                state,
                ctx_id,
                out_id,
                src.kind,
                [ow, src.ne[1], src.ne[2], src.ne[3]],
                OP_POOL_1D,
                &[Some(src.id)],
            )?;
            set_i32_params(state, out_id, &[(0, pop), (1, k0), (2, s0), (3, p0)])
        }
        "ggml_pool_2d" => {
            let (pop, k0, k1, s0, s1) = (
                arg_i32(&args[2])?,
                arg_i32(&args[3])?,
                arg_i32(&args[4])?,
                arg_i32(&args[5])?,
                arg_i32(&args[6])?,
            );
            let (p0, p1) = (arg_f32(&args[7])?, arg_f32(&args[8])?);
            let ow = (src.ne[0] + 2 * f32_to_i64_trunc(p0) - i64::from(k0)) / i64::from(s0) + 1;
            let oh = (src.ne[1] + 2 * f32_to_i64_trunc(p1) - i64::from(k1)) / i64::from(s1) + 1;
            finish_op(
                state,
                ctx_id,
                out_id,
                src.kind,
                [ow, oh, src.ne[2], src.ne[3]],
                OP_POOL_2D,
                &[Some(src.id)],
            )?;
            set_i32_params(
                state,
                out_id,
                &[(0, pop), (1, k0), (2, k1), (3, s0), (4, s1)],
            )?;
            set_f32_params(state, out_id, &[(5, p0), (6, p1)])
        }
        "ggml_upscale" => {
            let factor = i64::from(arg_i32(&args[2])?);
            finish_op(
                state,
                ctx_id,
                out_id,
                src.kind,
                [src.ne[0] * factor, src.ne[1] * factor, src.ne[2], src.ne[3]],
                OP_UPSCALE,
                &[Some(src.id)],
            )?;
            set_i32_params(state, out_id, &[(0, arg_i32(&args[3])?)])
        }
        _ => {
            finish_op(
                state,
                ctx_id,
                out_id,
                src.kind,
                [
                    arg_i64(&args[2])?,
                    arg_i64(&args[3])?,
                    arg_i64(&args[4])?,
                    arg_i64(&args[5])?,
                ],
                OP_UPSCALE,
                &[Some(src.id)],
            )?;
            set_i32_params(state, out_id, &[(0, arg_i32(&args[6])?)])
        }
    }
}

fn dispatch_pad_roll_tstep(
    state: &mut BackendState,
    func: &str,
    args: &[Value],
    res: &Value,
) -> Result<()> {
    let (ctx_id, out_id, src) = (
        arg_handle(&args[0], "Context")?,
        res_tensor(res)?,
        t_meta(state, args, 1)?,
    );
    match func {
        "ggml_pad" | "ggml_pad_circular" => {
            let p = [
                arg_i32(&args[2])?,
                arg_i32(&args[3])?,
                arg_i32(&args[4])?,
                arg_i32(&args[5])?,
            ];
            let dims = [
                src.ne[0] + i64::from(p[0]),
                src.ne[1] + i64::from(p[1]),
                src.ne[2] + i64::from(p[2]),
                src.ne[3] + i64::from(p[3]),
            ];
            finish_op(
                state,
                ctx_id,
                out_id,
                src.kind,
                dims,
                OP_PAD,
                &[Some(src.id)],
            )?;
            set_i32_params(
                state,
                out_id,
                &[
                    (0, 0),
                    (1, p[0]),
                    (2, 0),
                    (3, p[1]),
                    (4, 0),
                    (5, p[2]),
                    (6, 0),
                    (7, p[3]),
                    (8, i32::from(func == "ggml_pad_circular")),
                ],
            )
        }
        "ggml_pad_ext" | "ggml_pad_ext_circular" => {
            let v = (2..10)
                .map(|i| arg_i32(&args[i]))
                .collect::<Result<Vec<_>>>()?;
            let dims = [
                src.ne[0] + i64::from(v[0] + v[1]),
                src.ne[1] + i64::from(v[2] + v[3]),
                src.ne[2] + i64::from(v[4] + v[5]),
                src.ne[3] + i64::from(v[6] + v[7]),
            ];
            finish_op(
                state,
                ctx_id,
                out_id,
                src.kind,
                dims,
                OP_PAD,
                &[Some(src.id)],
            )?;
            let mut indexed: Vec<(usize, i32)> = v.into_iter().enumerate().collect();
            indexed.push((8, i32::from(func == "ggml_pad_ext_circular")));
            set_i32_params(state, out_id, &indexed)
        }
        _ => dispatch_reflect_roll_tstep(state, ctx_id, out_id, func, &src, args),
    }
}

fn dispatch_reflect_roll_tstep(
    state: &mut BackendState,
    ctx_id: u32,
    out_id: u32,
    func: &str,
    src: &TensorMeta,
    args: &[Value],
) -> Result<()> {
    match func {
        "ggml_pad_reflect_1d" => {
            let (lp0, rp0) = (arg_i32(&args[2])?, arg_i32(&args[3])?);
            finish_op(
                state,
                ctx_id,
                out_id,
                src.kind,
                [
                    src.ne[0] + i64::from(lp0 + rp0),
                    src.ne[1],
                    src.ne[2],
                    src.ne[3],
                ],
                OP_PAD_REFLECT_1D,
                &[Some(src.id)],
            )?;
            set_i32_params(state, out_id, &[(0, lp0), (1, rp0)])
        }
        "ggml_roll" => {
            finish_op(
                state,
                ctx_id,
                out_id,
                src.kind,
                src.ne,
                OP_ROLL,
                &[Some(src.id)],
            )?;
            set_i32_params(
                state,
                out_id,
                &[
                    (0, arg_i32(&args[2])?),
                    (1, arg_i32(&args[3])?),
                    (2, arg_i32(&args[4])?),
                    (3, arg_i32(&args[5])?),
                ],
            )
        }
        _ => {
            let dim = i64::from(arg_i32(&args[2])?);
            finish_op(
                state,
                ctx_id,
                out_id,
                TYPE_F32,
                [dim, src.ne[0], 1, 1],
                OP_TIMESTEP_EMBEDDING,
                &[Some(src.id)],
            )?;
            set_i32_params(
                state,
                out_id,
                &[(0, i32::try_from(dim)?), (1, arg_i32(&args[3])?)],
            )
        }
    }
}

fn dispatch_ssm_and_gla(
    state: &mut BackendState,
    func: &str,
    args: &[Value],
    res: &Value,
) -> Result<()> {
    let (ctx_id, out_id) = (arg_handle(&args[0], "Context")?, res_tensor(res)?);
    match func {
        "ggml_ssm_conv" => {
            let (sx, c_t) = (t_meta(state, args, 1)?, t_meta(state, args, 2)?);
            finish_op(
                state,
                ctx_id,
                out_id,
                TYPE_F32,
                [sx.ne[0], sx.ne[1] - c_t.ne[0] + 1, sx.ne[2], sx.ne[3]],
                OP_SSM_CONV,
                &[Some(sx.id), Some(c_t.id)],
            )
        }
        "ggml_ssm_scan" => {
            let (st, x_t) = (t_meta(state, args, 1)?, t_meta(state, args, 2)?);
            let elems =
                x_t.ne[0] * x_t.ne[1] * x_t.ne[2] + st.ne[0] * st.ne[1] * st.ne[2] * x_t.ne[2];
            finish_op(
                state,
                ctx_id,
                out_id,
                TYPE_F32,
                [elems, 1, 1, 1],
                OP_SSM_SCAN,
                &[
                    Some(st.id),
                    Some(x_t.id),
                    Some(t_id(args, 3)?),
                    Some(t_id(args, 4)?),
                    Some(t_id(args, 5)?),
                    Some(t_id(args, 6)?),
                    Some(t_id(args, 7)?),
                ],
            )?;
            set_i32_params(state, out_id, &[(0, i32::try_from(arg_i64(&args[8])?)?)])
        }
        "ggml_gated_linear_attn" => {
            let (v_t, st) = (t_meta(state, args, 2)?, t_meta(state, args, 5)?);
            finish_op(
                state,
                ctx_id,
                out_id,
                TYPE_F32,
                [v_t.ne[0] * v_t.ne[1] * v_t.ne[2] + st.nelements(), 1, 1, 1],
                OP_GATED_LINEAR_ATTN,
                &[
                    Some(t_id(args, 1)?),
                    Some(v_t.id),
                    Some(t_id(args, 3)?),
                    Some(t_id(args, 4)?),
                    Some(st.id),
                ],
            )?;
            set_f32_params(state, out_id, &[(0, arg_f32(&args[6])?)])
        }
        "ggml_rwkv_wkv6" => {
            let (v_t, st) = (t_meta(state, args, 2)?, t_meta(state, args, 6)?);
            finish_op(
                state,
                ctx_id,
                out_id,
                TYPE_F32,
                [v_t.ne[0] * v_t.ne[1] * v_t.ne[2] + st.nelements(), 1, 1, 1],
                OP_RWKV_WKV6,
                &[
                    Some(t_id(args, 1)?),
                    Some(v_t.id),
                    Some(t_id(args, 3)?),
                    Some(t_id(args, 4)?),
                    Some(t_id(args, 5)?),
                    Some(st.id),
                ],
            )
        }
        _ => {
            let (v_t, st) = (t_meta(state, args, 4)?, t_meta(state, args, 7)?);
            finish_op(
                state,
                ctx_id,
                out_id,
                TYPE_F32,
                [v_t.ne[0] * v_t.ne[1] * v_t.ne[2] + st.nelements(), 1, 1, 1],
                OP_RWKV_WKV7,
                &[
                    Some(t_id(args, 1)?),
                    Some(t_id(args, 2)?),
                    Some(t_id(args, 3)?),
                    Some(v_t.id),
                    Some(t_id(args, 5)?),
                    Some(t_id(args, 6)?),
                    Some(st.id),
                ],
            )
        }
    }
}

fn dispatch_gdn_ops(
    state: &mut BackendState,
    func: &str,
    args: &[Value],
    res: &Value,
) -> Result<()> {
    let (ctx_id, out_id, v_t, st) = (
        arg_handle(&args[0], "Context")?,
        res_tensor(res)?,
        t_meta(state, args, 3)?,
        t_meta(state, args, 6)?,
    );
    let base_elems = v_t.ne[0] * v_t.ne[1] * v_t.ne[2];
    match func {
        "ggml_gated_delta_net" => {
            finish_op(
                state,
                ctx_id,
                out_id,
                TYPE_F32,
                [
                    base_elems + v_t.ne[0] * v_t.ne[0] * v_t.ne[1] * st.ne[3],
                    1,
                    1,
                    1,
                ],
                OP_GATED_DELTA_NET,
                &[
                    Some(t_id(args, 1)?),
                    Some(t_id(args, 2)?),
                    Some(v_t.id),
                    Some(t_id(args, 4)?),
                    Some(t_id(args, 5)?),
                    Some(st.id),
                ],
            )?;
            set_i32_params(state, out_id, &[(0, i32::try_from(arg_i64(&args[7])?)?)])
        }
        "ggml_gated_delta_net_rows" => {
            let n_snap = arg_i32(&args[8])?;
            finish_op(
                state,
                ctx_id,
                out_id,
                TYPE_F32,
                [
                    base_elems + st.nelements() * i64::from(n_snap.max(1)),
                    1,
                    1,
                    1,
                ],
                OP_GATED_DELTA_NET,
                &[
                    Some(t_id(args, 1)?),
                    Some(t_id(args, 2)?),
                    Some(v_t.id),
                    Some(t_id(args, 4)?),
                    Some(t_id(args, 5)?),
                    Some(st.id),
                    Some(t_id(args, 7)?),
                ],
            )?;
            set_i32_params(state, out_id, &[(0, 0), (1, n_snap), (2, 1)])
        }
        _ => {
            finish_op(
                state,
                ctx_id,
                out_id,
                TYPE_F32,
                [
                    base_elems + v_t.ne[0] * v_t.ne[0] * v_t.ne[1] * st.ne[3],
                    1,
                    1,
                    1,
                ],
                OP_GATED_DELTA_NET_REPLAY,
                &[
                    Some(t_id(args, 1)?),
                    Some(t_id(args, 2)?),
                    Some(v_t.id),
                    Some(t_id(args, 4)?),
                    Some(t_id(args, 5)?),
                    Some(st.id),
                    Some(t_id(args, 7)?),
                    Some(t_id(args, 8)?),
                ],
            )?;
            set_i32_params(state, out_id, &[(0, arg_i32(&args[9])?)])
        }
    }
}

fn dispatch_misc_loss_opt(
    state: &mut BackendState,
    func: &str,
    args: &[Value],
    res: &Value,
) -> Result<()> {
    let (ctx_id, out_id) = (arg_handle(&args[0], "Context")?, res_tensor(res)?);
    match func {
        "ggml_solve_tri" => {
            let rhs = t_meta(state, args, 2)?;
            finish_op(
                state,
                ctx_id,
                out_id,
                TYPE_F32,
                rhs.ne,
                OP_SOLVE_TRI,
                &[Some(t_id(args, 1)?), Some(rhs.id)],
            )?;
            set_i32_params(
                state,
                out_id,
                &[
                    (0, i32::from(arg_bool(&args[3])?)),
                    (1, i32::from(arg_bool(&args[4])?)),
                    (2, i32::from(arg_bool(&args[5])?)),
                ],
            )
        }
        "ggml_lightning_indexer" => {
            let (q, k) = (t_meta(state, args, 1)?, t_meta(state, args, 2)?);
            finish_op(
                state,
                ctx_id,
                out_id,
                TYPE_F32,
                [k.ne[1], q.ne[1], q.ne[3], 1],
                OP_LIGHTNING_INDEXER,
                &[
                    Some(q.id),
                    Some(k.id),
                    Some(t_id(args, 3)?),
                    arg_opt_tensor(&args[4])?,
                ],
            )
        }
        "ggml_dsv4_hc_comb" => {
            let mixes = t_meta(state, args, 1)?;
            finish_op(
                state,
                ctx_id,
                out_id,
                TYPE_F32,
                mixes.ne,
                OP_DSV4_HC_COMB,
                &[Some(mixes.id), Some(t_id(args, 2)?), Some(t_id(args, 3)?)],
            )?;
            set_f32_params(state, out_id, &[(0, arg_f32(&args[4])?)])?;
            set_i32_params(state, out_id, &[(1, arg_i32(&args[5])?)])
        }
        "ggml_dsv4_hc_pre" => {
            let x = t_meta(state, args, 1)?;
            finish_op(
                state,
                ctx_id,
                out_id,
                TYPE_F32,
                [x.ne[0], x.ne[2], 1, 1],
                OP_DSV4_HC_PRE,
                &[Some(x.id), Some(t_id(args, 2)?)],
            )
        }
        "ggml_dsv4_hc_post" => {
            let x = t_meta(state, args, 1)?;
            finish_op(
                state,
                ctx_id,
                out_id,
                TYPE_F32,
                x.ne,
                OP_DSV4_HC_POST,
                &[
                    Some(x.id),
                    Some(t_id(args, 2)?),
                    Some(t_id(args, 3)?),
                    Some(t_id(args, 4)?),
                ],
            )
        }
        _ => dispatch_loss_and_opt(state, ctx_id, out_id, func, args),
    }
}

fn dispatch_loss_and_opt(
    state: &mut BackendState,
    ctx_id: u32,
    out_id: u32,
    func: &str,
    args: &[Value],
) -> Result<()> {
    match func {
        "ggml_cross_entropy_loss" => finish_op(
            state,
            ctx_id,
            out_id,
            TYPE_F32,
            [1, 1, 1, 1],
            OP_CROSS_ENTROPY_LOSS,
            &[Some(t_id(args, 1)?), Some(t_id(args, 2)?)],
        ),
        "ggml_cross_entropy_loss_back" => {
            let grad = t_meta(state, args, 1)?;
            finish_op(
                state,
                ctx_id,
                out_id,
                TYPE_F32,
                grad.ne,
                OP_CROSS_ENTROPY_LOSS_BACK,
                &[Some(grad.id), Some(t_id(args, 2)?), Some(t_id(args, 3)?)],
            )
        }
        "ggml_opt_step_adamw" => finish_op(
            state,
            ctx_id,
            out_id,
            TYPE_F32,
            [1, 1, 1, 1],
            OP_OPT_STEP_ADAMW,
            &[
                Some(t_id(args, 1)?),
                Some(t_id(args, 2)?),
                Some(t_id(args, 3)?),
                Some(t_id(args, 4)?),
                Some(t_id(args, 5)?),
            ],
        ),
        _ => finish_op(
            state,
            ctx_id,
            out_id,
            TYPE_F32,
            [1, 1, 1, 1],
            OP_OPT_STEP_SGD,
            &[
                Some(t_id(args, 1)?),
                Some(t_id(args, 2)?),
                Some(t_id(args, 3)?),
            ],
        ),
    }
}

pub(super) fn dispatch(
    state: &mut BackendState,
    func: &str,
    args: &[Value],
    res: &Value,
) -> Result<()> {
    if is_scalar_query(func) {
        return dispatch_query(state, func, args, res);
    }
    if matches!(
        func,
        "ggml_new_graph"
            | "ggml_new_graph_custom"
            | "ggml_build_forward_expand"
            | "ggml_build_backward_expand"
            | "ggml_graph_add_node"
            | "ggml_graph_cpy"
            | "ggml_graph_reset"
            | "ggml_backend_graph_compute"
    ) {
        return dispatch_graph_ops(state, func, args, res);
    }
    if matches!(
        func,
        "ggml_backend_buffer_set_usage"
            | "ggml_set_name"
            | "ggml_get_tensor"
            | "ggml_set_param"
            | "ggml_set_loss"
            | "ggml_mul_mat_set_hint"
            | "ggml_flash_attn_ext_add_sinks"
            | "ggml_flash_attn_ext_set_prec"
            | "ggml_soft_max_add_sinks"
            | "ggml_gated_delta_net_set_raw_gates"
            | "ggml_rope_set_offset"
    ) {
        return dispatch_meta_ops(state, func, args, res);
    }
    if func.starts_with("ggml_new_tensor") {
        return dispatch_new_tensors(state, func, args, res);
    }
    if func.starts_with("ggml_view_") {
        return dispatch_view_1_2(state, func, args, res);
    }
    if matches!(
        func,
        "ggml_reshape_2d"
            | "ggml_reshape_3d"
            | "ggml_reshape_4d"
            | "ggml_transpose"
            | "ggml_permute"
    ) {
        return dispatch_reshape_transpose_permute(state, func, args, res);
    }
    if func.contains("rope") {
        return dispatch_rope(state, func, args, res);
    }
    dispatch_family_two(state, func, args, res)
}

fn dispatch_family_two(
    state: &mut BackendState,
    func: &str,
    args: &[Value],
    res: &Value,
) -> Result<()> {
    if matches!(
        func,
        "ggml_norm"
            | "ggml_rms_norm"
            | "ggml_rms_norm_inplace"
            | "ggml_rms_norm_back"
            | "ggml_group_norm"
            | "ggml_l2_norm"
    ) {
        return dispatch_norm_ops(state, func, args, res);
    }
    if matches!(
        func,
        "ggml_mul_mat"
            | "ggml_mul_mat_id"
            | "ggml_out_prod"
            | "ggml_get_rows"
            | "ggml_get_rows_back"
            | "ggml_set_rows"
    ) {
        return dispatch_mat_and_rows(state, func, args, res);
    }
    if matches!(
        func,
        "ggml_soft_max"
            | "ggml_soft_max_ext"
            | "ggml_soft_max_ext_inplace"
            | "ggml_soft_max_ext_back"
            | "ggml_flash_attn_ext"
    ) {
        return dispatch_softmax_and_flash(state, func, args, res);
    }
    if matches!(
        func,
        "ggml_im2col"
            | "ggml_im2col_3d"
            | "ggml_conv_2d_direct"
            | "ggml_conv_2d_dw_direct"
            | "ggml_conv_3d_direct"
            | "ggml_conv_transpose_1d"
            | "ggml_conv_transpose_2d_p0"
    ) {
        return dispatch_conv_ops(state, func, args, res);
    }
    if matches!(
        func,
        "ggml_col2im_1d" | "ggml_pool_1d" | "ggml_pool_2d" | "ggml_upscale" | "ggml_interpolate"
    ) {
        return dispatch_pool_and_upscale(state, func, args, res);
    }
    dispatch_family_three(state, func, args, res)
}

fn dispatch_family_three(
    state: &mut BackendState,
    func: &str,
    args: &[Value],
    res: &Value,
) -> Result<()> {
    if func.starts_with("ggml_pad") || matches!(func, "ggml_roll" | "ggml_timestep_embedding") {
        return dispatch_pad_roll_tstep(state, func, args, res);
    }
    if matches!(
        func,
        "ggml_ssm_conv"
            | "ggml_ssm_scan"
            | "ggml_gated_linear_attn"
            | "ggml_rwkv_wkv6"
            | "ggml_rwkv_wkv7"
    ) {
        return dispatch_ssm_and_gla(state, func, args, res);
    }
    if func.starts_with("ggml_gated_delta_net") {
        return dispatch_gdn_ops(state, func, args, res);
    }
    if matches!(
        func,
        "ggml_solve_tri"
            | "ggml_lightning_indexer"
            | "ggml_dsv4_hc_comb"
            | "ggml_dsv4_hc_pre"
            | "ggml_dsv4_hc_post"
            | "ggml_cross_entropy_loss"
            | "ggml_cross_entropy_loss_back"
            | "ggml_opt_step_adamw"
            | "ggml_opt_step_sgd"
    ) {
        return dispatch_misc_loss_opt(state, func, args, res);
    }
    if matches!(
        func,
        "ggml_add"
            | "ggml_sub"
            | "ggml_mul"
            | "ggml_div"
            | "ggml_add_inplace"
            | "ggml_sub_inplace"
            | "ggml_mul_inplace"
            | "ggml_div_inplace"
            | "ggml_add_id"
            | "ggml_acc"
            | "ggml_set"
            | "ggml_set_inplace"
    ) {
        return dispatch_binary_arith(state, func, args, res);
    }
    if matches!(
        func,
        "ggml_cpy"
            | "ggml_dup"
            | "ggml_cont"
            | "ggml_repeat"
            | "ggml_repeat_4d"
            | "ggml_repeat_back"
            | "ggml_concat"
            | "ggml_count_equal"
    ) {
        return dispatch_copy_repeat_concat(state, func, args, res);
    }
    dispatch_unary_and_scalar_ops(state, func, args, res)
}
