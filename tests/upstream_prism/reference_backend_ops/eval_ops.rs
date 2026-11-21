//! Clean-room GGML graph evaluation and core tensor op kernels.
//!
//! No `bonsai_prism_runtime`, `bonsai_gguf`, or native GGML libraries are linked or called.

use super::eval_ops_extra::{
    eval_flash_attn, eval_im2col_and_conv, eval_opt_step, eval_pool, eval_rope, eval_solve_tri,
    eval_ssm_conv, eval_upscale_pad_roll, gelu_erf, write_flat_f32,
};
use super::state::BackendState;
use super::types::{
    FLAG_LOSS, OP_ACC, OP_ADD, OP_ADD_ID, OP_ARANGE, OP_ARGMAX, OP_ARGSORT, OP_CLAMP, OP_COL2IM_1D,
    OP_CONCAT, OP_CONT, OP_CONV_2D, OP_CONV_2D_DW, OP_CONV_3D, OP_CONV_TRANSPOSE_1D,
    OP_CONV_TRANSPOSE_2D, OP_COS, OP_COUNT_EQUAL, OP_CPY, OP_CROSS_ENTROPY_LOSS,
    OP_CROSS_ENTROPY_LOSS_BACK, OP_CUMSUM, OP_DIAG, OP_DIAG_MASK_INF, OP_DIV, OP_DSV4_HC_COMB,
    OP_DSV4_HC_POST, OP_DSV4_HC_PRE, OP_DUP, OP_FILL, OP_FLASH_ATTN_EXT, OP_GATED_DELTA_NET,
    OP_GATED_DELTA_NET_REPLAY, OP_GATED_LINEAR_ATTN, OP_GET_ROWS, OP_GET_ROWS_BACK, OP_GLU,
    OP_GROUP_NORM, OP_IM2COL, OP_IM2COL_3D, OP_L2_NORM, OP_LEAKY_RELU, OP_LIGHTNING_INDEXER,
    OP_LOG, OP_MEAN, OP_MUL, OP_MUL_MAT, OP_MUL_MAT_ID, OP_NONE, OP_NORM, OP_OPT_STEP_ADAMW,
    OP_OPT_STEP_SGD, OP_OUT_PROD, OP_PAD, OP_PAD_REFLECT_1D, OP_PERMUTE, OP_POOL_1D, OP_POOL_2D,
    OP_REPEAT, OP_REPEAT_BACK, OP_RESHAPE, OP_RMS_NORM, OP_RMS_NORM_BACK, OP_ROLL, OP_ROPE,
    OP_ROPE_BACK, OP_RWKV_WKV6, OP_RWKV_WKV7, OP_SCALE, OP_SET, OP_SET_ROWS, OP_SILU_BACK, OP_SIN,
    OP_SOFT_MAX, OP_SOFT_MAX_BACK, OP_SOLVE_TRI, OP_SQR, OP_SQRT, OP_SSM_CONV, OP_SSM_SCAN, OP_SUB,
    OP_SUM, OP_SUM_ROWS, OP_TIMESTEP_EMBEDDING, OP_TOP_K, OP_TRANSPOSE, OP_TRI, OP_UNARY,
    OP_UPSCALE, OP_VIEW, SORT_DESC, TRI_LOWER, TRI_LOWER_DIAG, TRI_UPPER, TYPE_I8, TYPE_I16,
    TYPE_I32, TYPE_I64, UNARY_OP_ABS, UNARY_OP_CEIL, UNARY_OP_ELU, UNARY_OP_EXP, UNARY_OP_EXPM1,
    UNARY_OP_FLOOR, UNARY_OP_GELU, UNARY_OP_GELU_ERF, UNARY_OP_GELU_QUICK, UNARY_OP_HARDSIGMOID,
    UNARY_OP_HARDSWISH, UNARY_OP_NEG, UNARY_OP_RELU, UNARY_OP_ROUND, UNARY_OP_SGN,
    UNARY_OP_SIGMOID, UNARY_OP_SILU, UNARY_OP_SOFTPLUS, UNARY_OP_STEP, UNARY_OP_TANH,
    UNARY_OP_TRUNC, UNARY_OP_XIELU, blck_size, f64_to_f32, i32_to_f32, type_size, usize_to_f32,
};
use crate::{Result, invalid};

fn is_exact_int_type(kind: i32) -> bool {
    matches!(kind, TYPE_I8 | TYPE_I16 | TYPE_I32 | TYPE_I64)
}

fn eval_unary_scalar(uop: i32, x: f32, meta_params: &[f32; 4]) -> f32 {
    match uop {
        UNARY_OP_ABS => x.abs(),
        UNARY_OP_SGN => {
            if x > 0.0 {
                1.0
            } else if x < 0.0 {
                -1.0
            } else {
                0.0
            }
        }
        UNARY_OP_NEG => -x,
        UNARY_OP_STEP => {
            if x > 0.0 {
                1.0
            } else {
                0.0
            }
        }
        UNARY_OP_TANH => x.tanh(),
        UNARY_OP_ELU => {
            if x > 0.0 {
                x
            } else {
                x.exp_m1()
            }
        }
        UNARY_OP_RELU => x.max(0.0),
        UNARY_OP_SIGMOID => 1.0 / (1.0 + (-x).exp()),
        UNARY_OP_GELU => {
            let c = (2.0_f32 / std::f32::consts::PI).sqrt();
            0.5 * x * (1.0 + (c * (0.044_715 * x * x).mul_add(x, x)).tanh())
        }
        UNARY_OP_GELU_QUICK => x / (1.0 + (-1.702 * x).exp()),
        UNARY_OP_SILU => x / (1.0 + (-x).exp()),
        UNARY_OP_HARDSWISH => x * ((x + 3.0).clamp(0.0, 6.0) / 6.0),
        UNARY_OP_HARDSIGMOID => (x + 3.0).clamp(0.0, 6.0) / 6.0,
        UNARY_OP_EXP => x.exp(),
        UNARY_OP_EXPM1 => x.exp_m1(),
        UNARY_OP_SOFTPLUS => {
            if x > 20.0 {
                x
            } else {
                x.exp().ln_1p()
            }
        }
        UNARY_OP_GELU_ERF => gelu_erf(x),
        UNARY_OP_XIELU => {
            let [alpha_n, alpha_p, beta, eps] = *meta_params;
            if x > 0.0 {
                (alpha_p * x).mul_add(x, beta * x)
            } else {
                alpha_n.mul_add(x.min(eps).exp_m1() - x, beta * x)
            }
        }
        UNARY_OP_FLOOR => x.floor(),
        UNARY_OP_CEIL => x.ceil(),
        UNARY_OP_ROUND => x.round_ties_even(),
        UNARY_OP_TRUNC => x.trunc(),
        _ => x,
    }
}

fn eval_copy_like(state: &mut BackendState, tid: u32) -> Result<()> {
    let dst = state.tensor(tid)?.clone();
    let src_id = dst.sources[0].ok_or_else(|| invalid("copy op missing src"))?;
    let src = state.tensor(src_id)?.clone();
    if src.kind == dst.kind && blck_size(src.kind) == 1 && !is_exact_int_type(src.kind) {
        let ts = usize::try_from(type_size(src.kind))?;
        let (buf_id, base) = state.resolve_storage(src_id)?;
        let buf = state
            .buffers
            .get(&buf_id)
            .ok_or_else(|| invalid("copy src missing buffer"))?;
        let mut flat = Vec::with_capacity(usize::try_from(src.nelements())? * ts);
        for d3 in 0..src.ne[3] {
            for d2 in 0..src.ne[2] {
                for d1 in 0..src.ne[1] {
                    for d0 in 0..src.ne[0] {
                        let off = usize::try_from(
                            base + u64::try_from(d0)? * src.nb[0]
                                + u64::try_from(d1)? * src.nb[1]
                                + u64::try_from(d2)? * src.nb[2]
                                + u64::try_from(d3)? * src.nb[3],
                        )?;
                        let elem = buf
                            .get(off..off + ts)
                            .ok_or_else(|| invalid("copy raw read exceeds buffer"))?;
                        flat.extend_from_slice(elem);
                    }
                }
            }
        }
        let mut idx = 0_usize;
        for d3 in 0..dst.ne[3] {
            for d2 in 0..dst.ne[2] {
                for d1 in 0..dst.ne[1] {
                    for d0 in 0..dst.ne[0] {
                        let chunk = &flat[idx * ts..(idx + 1) * ts];
                        let rel = u64::try_from(d0)? * dst.nb[0]
                            + u64::try_from(d1)? * dst.nb[1]
                            + u64::try_from(d2)? * dst.nb[2]
                            + u64::try_from(d3)? * dst.nb[3];
                        state.write_raw(tid, rel, chunk)?;
                        idx += 1;
                    }
                }
            }
        }
        return Ok(());
    }
    if is_exact_int_type(src.kind) && is_exact_int_type(dst.kind) {
        let mut ints = Vec::with_capacity(usize::try_from(src.nelements())?);
        for d3 in 0..src.ne[3] {
            for d2 in 0..src.ne[2] {
                for d1 in 0..src.ne[1] {
                    for d0 in 0..src.ne[0] {
                        ints.push(state.read_elem_i32(src_id, [d0, d1, d2, d3])?);
                    }
                }
            }
        }
        let mut pos = 0_usize;
        for d3 in 0..dst.ne[3] {
            for d2 in 0..dst.ne[2] {
                for d1 in 0..dst.ne[1] {
                    for d0 in 0..dst.ne[0] {
                        state.write_elem_i32(tid, [d0, d1, d2, d3], ints[pos])?;
                        pos += 1;
                    }
                }
            }
        }
        return Ok(());
    }
    let (_, vals) = state.read_all_f32(src_id)?;
    write_flat_f32(state, tid, &vals)
}

fn eval_binary_bcast(state: &mut BackendState, tid: u32, op_code: i32) -> Result<()> {
    let dst = state.tensor(tid)?.clone();
    let lhs_id = dst.sources[0].ok_or_else(|| invalid("binary missing lhs"))?;
    let rhs_id = dst.sources[1].ok_or_else(|| invalid("binary missing rhs"))?;
    let lhs = state.tensor(lhs_id)?.clone();
    let rhs = state.tensor(rhs_id)?.clone();
    let mut rows = Vec::new();
    for d3 in 0..dst.ne[3] {
        for d2 in 0..dst.ne[2] {
            for d1 in 0..dst.ne[1] {
                let l_row =
                    state.read_row_f32(lhs_id, d1 % lhs.ne[1], d2 % lhs.ne[2], d3 % lhs.ne[3])?;
                let r_row =
                    state.read_row_f32(rhs_id, d1 % rhs.ne[1], d2 % rhs.ne[2], d3 % rhs.ne[3])?;
                let n0 = usize::try_from(dst.ne[0])?;
                let mut out_row = Vec::with_capacity(n0);
                for idx in 0..n0 {
                    let a_val = l_row[idx % l_row.len()];
                    let b_val = r_row[idx % r_row.len()];
                    let res = match op_code {
                        OP_ADD => a_val + b_val,
                        OP_SUB => a_val - b_val,
                        OP_MUL => a_val * b_val,
                        OP_DIV => a_val / b_val,
                        _ => a_val,
                    };
                    out_row.push(res);
                }
                rows.push((d1, d2, d3, out_row));
            }
        }
    }
    for (d1, d2, d3, out_row) in rows {
        state.write_row_f32(tid, d1, d2, d3, &out_row)?;
    }
    Ok(())
}

fn eval_get_rows(state: &mut BackendState, tid: u32) -> Result<()> {
    let dst = state.tensor(tid)?.clone();
    let src_id = dst.sources[0].ok_or_else(|| invalid("get_rows missing src"))?;
    let rows_id = dst.sources[1].ok_or_else(|| invalid("get_rows missing ids"))?;
    let src = state.tensor(src_id)?.clone();
    let rows_meta = state.tensor(rows_id)?.clone();
    let mut float_writes = Vec::new();
    let mut int_writes = Vec::new();
    for d3 in 0..rows_meta.ne[2] {
        for d2 in 0..rows_meta.ne[1] {
            for d1 in 0..rows_meta.ne[0] {
                let row_idx = i64::from(state.read_elem_i32(rows_id, [d1, d2, d3, 0])?);
                if src.kind == TYPE_I32 && dst.kind == TYPE_I32 {
                    let mut out_row = Vec::with_capacity(usize::try_from(dst.ne[0])?);
                    for d0 in 0..src.ne[0] {
                        out_row.push(state.read_elem_i32(
                            src_id,
                            [d0, row_idx, d2 % src.ne[2], d3 % src.ne[3]],
                        )?);
                    }
                    int_writes.push((d1, d2, d3, out_row));
                } else {
                    let out_row =
                        state.read_row_f32(src_id, row_idx, d2 % src.ne[2], d3 % src.ne[3])?;
                    float_writes.push((d1, d2, d3, out_row));
                }
            }
        }
    }
    for (d1, d2, d3, row) in float_writes {
        state.write_row_f32(tid, d1, d2, d3, &row)?;
    }
    for (d1, d2, d3, row) in int_writes {
        for (d0, val) in row.into_iter().enumerate() {
            state.write_elem_i32(tid, [i64::try_from(d0)?, d1, d2, d3], val)?;
        }
    }
    Ok(())
}

fn eval_set_rows(state: &mut BackendState, tid: u32) -> Result<()> {
    let dst = state.tensor(tid)?.clone();
    let src_id = dst.sources[0].ok_or_else(|| invalid("set_rows missing src"))?;
    let rows_id = dst.sources[1].ok_or_else(|| invalid("set_rows missing rows"))?;
    let src = state.tensor(src_id)?.clone();
    for d3 in 0..src.ne[3] {
        for d2 in 0..src.ne[2] {
            for d1 in 0..src.ne[1] {
                let target_row = i64::from(state.read_elem_i32(rows_id, [d1, d2, d3, 0])?);
                let vals = state.read_row_f32(src_id, d1, d2, d3)?;
                state.write_row_f32(tid, target_row, d2, d3, &vals)?;
            }
        }
    }
    Ok(())
}

fn eval_mul_mat(state: &mut BackendState, tid: u32) -> Result<()> {
    let dst = state.tensor(tid)?.clone();
    let a_id = dst.sources[0].ok_or_else(|| invalid("mul_mat missing a"))?;
    let b_id = dst.sources[1].ok_or_else(|| invalid("mul_mat missing b"))?;
    let a_meta = state.tensor(a_id)?.clone();
    let b_meta = state.tensor(b_id)?.clone();
    let r2 = (dst.ne[2] / a_meta.ne[2].max(1)).max(1);
    let r3 = (dst.ne[3] / a_meta.ne[3].max(1)).max(1);
    let mut rows = Vec::new();
    for d3 in 0..dst.ne[3] {
        let (a3, b3) = (d3 / r3, d3 % b_meta.ne[3]);
        for d2 in 0..dst.ne[2] {
            let (a2, b2) = (d2 / r2, d2 % b_meta.ne[2]);
            let mut a_rows = Vec::with_capacity(usize::try_from(a_meta.ne[1])?);
            for col in 0..a_meta.ne[1] {
                a_rows.push(state.read_row_f32(a_id, col, a2, a3)?);
            }
            for row_n in 0..dst.ne[1] {
                let b_row = state.read_row_f32(b_id, row_n, b2, b3)?;
                let mut out_row = Vec::with_capacity(a_rows.len());
                for a_row in &a_rows {
                    let sum: f64 = a_row.iter().zip(b_row.iter()).fold(0.0, |acc, (&av, &bv)| {
                        f64::from(av).mul_add(f64::from(bv), acc)
                    });
                    out_row.push(f64_to_f32(sum));
                }
                rows.push((row_n, d2, d3, out_row));
            }
        }
    }
    for (row_n, d2, d3, out_row) in rows {
        state.write_row_f32(tid, row_n, d2, d3, &out_row)?;
    }
    Ok(())
}

fn eval_mul_mat_id(state: &mut BackendState, tid: u32) -> Result<()> {
    let dst = state.tensor(tid)?.clone();
    let a_id = dst.sources[0].ok_or_else(|| invalid("mul_mat_id missing as"))?;
    let b_id = dst.sources[1].ok_or_else(|| invalid("mul_mat_id missing b"))?;
    let ids_id = dst.sources[2].ok_or_else(|| invalid("mul_mat_id missing ids"))?;
    let a_meta = state.tensor(a_id)?.clone();
    let b_meta = state.tensor(b_id)?.clone();
    let mut rows = Vec::new();
    for tok in 0..dst.ne[2] {
        for slot in 0..dst.ne[1] {
            let expert = i64::from(state.read_elem_i32(ids_id, [slot, tok, 0, 0])?);
            let b1 = if b_meta.ne[1] == 1 { 0 } else { slot };
            let b_row = state.read_row_f32(b_id, b1, tok, 0)?;
            let mut out_row = Vec::with_capacity(usize::try_from(a_meta.ne[1])?);
            for col in 0..a_meta.ne[1] {
                let a_row = state.read_row_f32(a_id, col, expert, 0)?;
                let sum: f64 = a_row.iter().zip(b_row.iter()).fold(0.0, |acc, (&av, &bv)| {
                    f64::from(av).mul_add(f64::from(bv), acc)
                });
                out_row.push(f64_to_f32(sum));
            }
            rows.push((slot, tok, 0, out_row));
        }
    }
    for (slot, tok, d3, out_row) in rows {
        state.write_row_f32(tid, slot, tok, d3, &out_row)?;
    }
    Ok(())
}

fn eval_norms(state: &mut BackendState, tid: u32, op_code: i32) -> Result<()> {
    let dst = state.tensor(tid)?.clone();
    let src_id = dst.sources[0].ok_or_else(|| invalid("norm missing src"))?;
    if op_code == OP_GROUP_NORM {
        let n_groups = i64::from(dst.op_param_i32(0).max(1));
        let eps = dst.op_param_f32(1);
        let group_size = dst.ne[2] / n_groups;
        let (_, src_vals) = state.read_all_f32(src_id)?;
        let plane = usize::try_from(dst.ne[0] * dst.ne[1])?;
        let mut out = vec![0.0_f32; src_vals.len()];
        for batch in 0..usize::try_from(dst.ne[3])? {
            let b_off = batch * usize::try_from(dst.ne[2])? * plane;
            for grp in 0..usize::try_from(n_groups)? {
                let g_start = b_off + grp * usize::try_from(group_size)? * plane;
                let g_end = g_start + usize::try_from(group_size)? * plane;
                let slice = &src_vals[g_start..g_end];
                let mean = slice.iter().copied().map(f64::from).sum::<f64>()
                    / f64::from(i32::try_from(slice.len()).unwrap_or(1));
                let var = slice
                    .iter()
                    .copied()
                    .map(|v| (f64::from(v) - mean).powi(2))
                    .sum::<f64>()
                    / f64::from(i32::try_from(slice.len()).unwrap_or(1));
                let inv_std = 1.0 / (var + f64::from(eps)).sqrt();
                for (d, &s) in out[g_start..g_end].iter_mut().zip(slice.iter()) {
                    *d = f64_to_f32((f64::from(s) - mean) * inv_std);
                }
            }
        }
        return write_flat_f32(state, tid, &out);
    }
    let eps = dst.op_param_f32(0);
    let mut rows = Vec::new();
    for d3 in 0..dst.ne[3] {
        for d2 in 0..dst.ne[2] {
            for d1 in 0..dst.ne[1] {
                let row = state.read_row_f32(src_id, d1, d2, d3)?;
                let len_f = f64::from(i32::try_from(row.len()).unwrap_or(1));
                let out_row: Vec<f32> = if op_code == OP_NORM {
                    let mean = row.iter().copied().map(f64::from).sum::<f64>() / len_f;
                    let var = row
                        .iter()
                        .copied()
                        .map(|v| (f64::from(v) - mean).powi(2))
                        .sum::<f64>()
                        / len_f;
                    let inv = 1.0 / (var + f64::from(eps)).sqrt();
                    row.iter()
                        .map(|&v| f64_to_f32((f64::from(v) - mean) * inv))
                        .collect()
                } else if op_code == OP_RMS_NORM {
                    let ms = row
                        .iter()
                        .copied()
                        .map(|v| f64::from(v).powi(2))
                        .sum::<f64>()
                        / len_f;
                    let scale = 1.0 / (ms + f64::from(eps)).sqrt();
                    row.iter()
                        .map(|&v| f64_to_f32(f64::from(v) * scale))
                        .collect()
                } else {
                    let ss = row
                        .iter()
                        .copied()
                        .map(|v| f64::from(v).powi(2))
                        .sum::<f64>();
                    let scale = 1.0 / ss.sqrt().max(f64::from(eps));
                    row.iter()
                        .map(|&v| f64_to_f32(f64::from(v) * scale))
                        .collect()
                };
                rows.push((d1, d2, d3, out_row));
            }
        }
    }
    for (d1, d2, d3, out_row) in rows {
        state.write_row_f32(tid, d1, d2, d3, &out_row)?;
    }
    Ok(())
}

fn eval_softmax(state: &mut BackendState, tid: u32) -> Result<()> {
    let dst = state.tensor(tid)?.clone();
    let src_id = dst.sources[0].ok_or_else(|| invalid("softmax missing src"))?;
    let (mask_id, sinks_id) = (dst.sources[1], dst.sources[2]);
    let (scale, max_bias) = (dst.op_param_f32(0), dst.op_param_f32(1));
    let n_heads = dst.ne[2].max(1);
    let mut rows = Vec::new();
    for d3 in 0..dst.ne[3] {
        for d2 in 0..dst.ne[2] {
            let slope = if max_bias > 0.0 {
                let n_log2 = u32::try_from(n_heads).ok().map_or(0, u32::ilog2);
                let base_pow =
                    (-8.0 / i32_to_f32(i32::try_from(n_log2).unwrap_or(0)).exp2()).exp2();
                base_pow.powi(i32::try_from(d2 + 1).unwrap_or(1))
            } else {
                1.0
            };
            let sink = sinks_id
                .map(|sid| state.read_elem_f32(sid, [d2, 0, 0, 0]))
                .transpose()?;
            for d1 in 0..dst.ne[1] {
                let mut row = state.read_row_f32(src_id, d1, d2, d3)?;
                let mut max_v = sink.unwrap_or(f32::NEG_INFINITY);
                for (d0, val) in row.iter_mut().enumerate() {
                    let mut x = *val * scale;
                    if let Some(mid) = mask_id {
                        let mm = state.tensor(mid)?;
                        let mv = state.read_elem_f32(
                            mid,
                            [
                                i64::try_from(d0)?,
                                d1 % mm.ne[1],
                                d2 % mm.ne[2],
                                d3 % mm.ne[3],
                            ],
                        )?;
                        x = slope.mul_add(mv, x);
                    }
                    *val = x;
                    max_v = max_v.max(x);
                }
                let mut sum = sink.map_or(0.0, |s| (s - max_v).exp());
                for val in &mut row {
                    *val = (*val - max_v).exp();
                    sum += *val;
                }
                if sum > 0.0 {
                    for val in &mut row {
                        *val /= sum;
                    }
                }
                rows.push((d1, d2, d3, row));
            }
        }
    }
    for (d1, d2, d3, row) in rows {
        state.write_row_f32(tid, d1, d2, d3, &row)?;
    }
    Ok(())
}

fn eval_elementwise_misc(state: &mut BackendState, tid: u32, op_code: i32) -> Result<()> {
    let dst = state.tensor(tid)?.clone();
    match op_code {
        OP_FILL => write_flat_f32(
            state,
            tid,
            &vec![dst.op_param_f32(0); usize::try_from(dst.nelements())?],
        ),
        OP_ARANGE => {
            let (start, step) = (dst.op_param_f32(0), dst.op_param_f32(2));
            let vals: Vec<f32> = (0..usize::try_from(dst.ne[0])?)
                .map(|i| usize_to_f32(i).mul_add(step, start))
                .collect();
            write_flat_f32(state, tid, &vals)
        }
        OP_CLAMP => {
            let src_id = dst.sources[0].ok_or_else(|| invalid("clamp missing src"))?;
            let (lo, hi) = (dst.op_param_f32(0), dst.op_param_f32(1));
            let (_, mut vals) = state.read_all_f32(src_id)?;
            for v in &mut vals {
                *v = v.clamp(lo, hi);
            }
            write_flat_f32(state, tid, &vals)
        }
        OP_SCALE => {
            let src_id = dst.sources[0].ok_or_else(|| invalid("scale missing src"))?;
            let (s, b) = (dst.op_param_f32(0), dst.op_param_f32(1));
            let (_, mut vals) = state.read_all_f32(src_id)?;
            for v in &mut vals {
                *v = (*v).mul_add(s, b);
            }
            write_flat_f32(state, tid, &vals)
        }
        OP_LEAKY_RELU => {
            let src_id = dst.sources[0].ok_or_else(|| invalid("leaky_relu missing src"))?;
            let slope = dst.op_param_f32(0);
            let (_, mut vals) = state.read_all_f32(src_id)?;
            for v in &mut vals {
                if *v < 0.0 {
                    *v *= slope;
                }
            }
            write_flat_f32(state, tid, &vals)
        }
        OP_UNARY => {
            let src_id = dst.sources[0].ok_or_else(|| invalid("unary missing src"))?;
            let uop = dst.op_param_i32(0);
            let params = [
                dst.op_param_f32(1),
                dst.op_param_f32(2),
                dst.op_param_f32(3),
                dst.op_param_f32(4),
            ];
            let (_, mut vals) = state.read_all_f32(src_id)?;
            for v in &mut vals {
                *v = eval_unary_scalar(uop, *v, &params);
            }
            write_flat_f32(state, tid, &vals)
        }
        OP_SQR | OP_SQRT | OP_LOG | OP_SIN | OP_COS => {
            let src_id = dst.sources[0].ok_or_else(|| invalid("math_unary missing src"))?;
            let (_, mut vals) = state.read_all_f32(src_id)?;
            for v in &mut vals {
                *v = match op_code {
                    OP_SQR => *v * *v,
                    OP_SQRT => v.sqrt(),
                    OP_LOG => v.ln(),
                    OP_SIN => v.sin(),
                    _ => v.cos(),
                };
            }
            write_flat_f32(state, tid, &vals)
        }
        _ => Err(invalid(format!("unhandled elementwise misc op {op_code}"))),
    }
}

fn eval_sum_and_mean(state: &mut BackendState, tid: u32, op_code: i32) -> Result<()> {
    let dst = state.tensor(tid)?.clone();
    let src_id = dst.sources[0].ok_or_else(|| invalid("sum missing src"))?;
    if op_code == OP_SUM {
        let (_, vals) = state.read_all_f32(src_id)?;
        let total = f64_to_f32(vals.iter().copied().map(f64::from).sum::<f64>());
        return state.write_elem_f32(tid, [0, 0, 0, 0], total);
    }
    let mut rows = Vec::new();
    for d3 in 0..dst.ne[3] {
        for d2 in 0..dst.ne[2] {
            for d1 in 0..dst.ne[1] {
                let row = state.read_row_f32(src_id, d1, d2, d3)?;
                if op_code == OP_CUMSUM {
                    let mut acc = 0.0_f64;
                    let cum: Vec<f32> = row
                        .iter()
                        .map(|&v| {
                            acc += f64::from(v);
                            f64_to_f32(acc)
                        })
                        .collect();
                    rows.push((d1, d2, d3, cum));
                } else {
                    let sum = row.iter().copied().map(f64::from).sum::<f64>();
                    let res = if op_code == OP_MEAN {
                        sum / f64::from(i32::try_from(row.len()).unwrap_or(1))
                    } else {
                        sum
                    };
                    rows.push((d1, d2, d3, vec![f64_to_f32(res)]));
                }
            }
        }
    }
    for (d1, d2, d3, row) in rows {
        state.write_row_f32(tid, d1, d2, d3, &row)?;
    }
    Ok(())
}

fn eval_argmax_sort_count(state: &mut BackendState, tid: u32, op_code: i32) -> Result<()> {
    let dst = state.tensor(tid)?.clone();
    match op_code {
        OP_ARGMAX => {
            let src_id = dst.sources[0].ok_or_else(|| invalid("argmax missing src"))?;
            for d3 in 0..dst.ne[3] {
                for d2 in 0..dst.ne[2] {
                    for d1 in 0..dst.ne[1] {
                        let row = state.read_row_f32(src_id, d1, d2, d3)?;
                        let best_i = row
                            .iter()
                            .enumerate()
                            .max_by(|a, b| a.1.total_cmp(b.1))
                            .map_or(0, |(i, _)| i);
                        state.write_elem_i32(tid, [0, d1, d2, d3], i32::try_from(best_i)?)?;
                    }
                }
            }
            Ok(())
        }
        OP_ARGSORT | OP_TOP_K => {
            let src_id = dst.sources[0].ok_or_else(|| invalid("sort missing src"))?;
            let desc = dst.op_param_i32(0) == SORT_DESC || op_code == OP_TOP_K;
            for d3 in 0..dst.ne[3] {
                for d2 in 0..dst.ne[2] {
                    for d1 in 0..dst.ne[1] {
                        let row = state.read_row_f32(src_id, d1, d2, d3)?;
                        let mut indices: Vec<usize> = (0..row.len()).collect();
                        indices.sort_by(|&a, &b| {
                            let ord = row[a].total_cmp(&row[b]);
                            (if desc { ord.reverse() } else { ord }).then_with(|| a.cmp(&b))
                        });
                        for (k, &idx_val) in
                            indices.iter().take(usize::try_from(dst.ne[0])?).enumerate()
                        {
                            state.write_elem_i32(
                                tid,
                                [i64::try_from(k)?, d1, d2, d3],
                                i32::try_from(idx_val)?,
                            )?;
                        }
                    }
                }
            }
            Ok(())
        }
        OP_COUNT_EQUAL => {
            let (a_id, b_id) = (
                dst.sources[0].ok_or_else(|| invalid("count_equal a"))?,
                dst.sources[1].ok_or_else(|| invalid("count_equal b"))?,
            );
            let a = state.tensor(a_id)?.clone();
            let mut count = 0_i64;
            for d3 in 0..a.ne[3] {
                for d2 in 0..a.ne[2] {
                    for d1 in 0..a.ne[1] {
                        for d0 in 0..a.ne[0] {
                            if state.read_elem_i32(a_id, [d0, d1, d2, d3])?
                                == state.read_elem_i32(b_id, [d0, d1, d2, d3])?
                            {
                                count += 1;
                            }
                        }
                    }
                }
            }
            state.write_raw(tid, 0, &count.to_le_bytes())
        }
        _ => Err(invalid(format!("unhandled index op {op_code}"))),
    }
}

fn eval_glu(state: &mut BackendState, tid: u32) -> Result<()> {
    let dst = state.tensor(tid)?.clone();
    let a_id = dst.sources[0].ok_or_else(|| invalid("glu missing src0"))?;
    let (gop, swapped) = (dst.op_param_i32(0), dst.op_param_i32(1) != 0);
    let (alpha, limit) = (dst.op_param_f32(2), dst.op_param_f32(3));
    let half = usize::try_from(dst.ne[0])?;
    let mut rows = Vec::new();
    for d3 in 0..dst.ne[3] {
        for d2 in 0..dst.ne[2] {
            for d1 in 0..dst.ne[1] {
                let (gate_slice, val_slice) = if let Some(b_id) = dst.sources[1] {
                    (
                        state.read_row_f32(a_id, d1, d2, d3)?,
                        state.read_row_f32(b_id, d1, d2, d3)?,
                    )
                } else {
                    let full = state.read_row_f32(a_id, d1, d2, d3)?;
                    if swapped {
                        (full[half..half * 2].to_vec(), full[0..half].to_vec())
                    } else {
                        (full[0..half].to_vec(), full[half..half * 2].to_vec())
                    }
                };
                let mut out_row = Vec::with_capacity(half);
                for (&gv, &xv) in gate_slice.iter().zip(val_slice.iter()) {
                    let y = match gop {
                        0 => gv.max(0.0) * xv,
                        1 => eval_unary_scalar(UNARY_OP_GELU, gv, &[0.0; 4]) * xv,
                        2 => eval_unary_scalar(UNARY_OP_SILU, gv, &[0.0; 4]) * xv,
                        3 => {
                            let xg = gv.min(limit);
                            let xi = xv.clamp(-limit, limit);
                            (xg / (1.0 + (-xg * alpha).exp())) * (xi + 1.0)
                        }
                        4 => gelu_erf(gv) * xv,
                        _ => eval_unary_scalar(UNARY_OP_GELU_QUICK, gv, &[0.0; 4]) * xv,
                    };
                    out_row.push(y);
                }
                rows.push((d1, d2, d3, out_row));
            }
        }
    }
    for (d1, d2, d3, row) in rows {
        state.write_row_f32(tid, d1, d2, d3, &row)?;
    }
    Ok(())
}

fn eval_repeat_concat(state: &mut BackendState, tid: u32, is_concat: bool) -> Result<()> {
    let dst = state.tensor(tid)?.clone();
    if is_concat {
        let (a_id, b_id) = (
            dst.sources[0].ok_or_else(|| invalid("concat a"))?,
            dst.sources[1].ok_or_else(|| invalid("concat b"))?,
        );
        let a = state.tensor(a_id)?.clone();
        let dim = dst.op_param_i32(0);
        for d3 in 0..dst.ne[3] {
            for d2 in 0..dst.ne[2] {
                for d1 in 0..dst.ne[1] {
                    for d0 in 0..dst.ne[0] {
                        let val = match dim {
                            0 if d0 < a.ne[0] => state.read_elem_f32(a_id, [d0, d1, d2, d3])?,
                            0 => state.read_elem_f32(b_id, [d0 - a.ne[0], d1, d2, d3])?,
                            1 if d1 < a.ne[1] => state.read_elem_f32(a_id, [d0, d1, d2, d3])?,
                            1 => state.read_elem_f32(b_id, [d0, d1 - a.ne[1], d2, d3])?,
                            2 if d2 < a.ne[2] => state.read_elem_f32(a_id, [d0, d1, d2, d3])?,
                            2 => state.read_elem_f32(b_id, [d0, d1, d2 - a.ne[2], d3])?,
                            _ if d3 < a.ne[3] => state.read_elem_f32(a_id, [d0, d1, d2, d3])?,
                            _ => state.read_elem_f32(b_id, [d0, d1, d2, d3 - a.ne[3]])?,
                        };
                        state.write_elem_f32(tid, [d0, d1, d2, d3], val)?;
                    }
                }
            }
        }
        return Ok(());
    }
    let src_id = dst.sources[0].ok_or_else(|| invalid("repeat missing src"))?;
    let src = state.tensor(src_id)?.clone();
    let mut rows = Vec::new();
    for d3 in 0..dst.ne[3] {
        for d2 in 0..dst.ne[2] {
            for d1 in 0..dst.ne[1] {
                let s_row =
                    state.read_row_f32(src_id, d1 % src.ne[1], d2 % src.ne[2], d3 % src.ne[3])?;
                let mut out_row = Vec::with_capacity(usize::try_from(dst.ne[0])?);
                for d0 in 0..usize::try_from(dst.ne[0])? {
                    out_row.push(s_row[d0 % s_row.len()]);
                }
                rows.push((d1, d2, d3, out_row));
            }
        }
    }
    for (d1, d2, d3, row) in rows {
        state.write_row_f32(tid, d1, d2, d3, &row)?;
    }
    Ok(())
}

fn eval_diag_tri(state: &mut BackendState, tid: u32, op_code: i32) -> Result<()> {
    let dst = state.tensor(tid)?.clone();
    let src_id = dst.sources[0].ok_or_else(|| invalid("diag/tri missing src"))?;
    match op_code {
        OP_DIAG_MASK_INF => {
            let n_past = i64::from(dst.op_param_i32(0));
            for d3 in 0..dst.ne[3] {
                for d2 in 0..dst.ne[2] {
                    for d1 in 0..dst.ne[1] {
                        let mut row = state.read_row_f32(src_id, d1, d2, d3)?;
                        for (d0, val) in row.iter_mut().enumerate() {
                            if i64::try_from(d0)? > n_past + d1 {
                                *val = f32::NEG_INFINITY;
                            }
                        }
                        state.write_row_f32(tid, d1, d2, d3, &row)?;
                    }
                }
            }
        }
        OP_TRI => {
            let tri_kind = dst.op_param_i32(0);
            for d3 in 0..dst.ne[3] {
                for d2 in 0..dst.ne[2] {
                    for d1 in 0..dst.ne[1] {
                        let mut row = state.read_row_f32(src_id, d1, d2, d3)?;
                        for (d0, val) in row.iter_mut().enumerate() {
                            let col = i64::try_from(d0)?;
                            let keep = match tri_kind {
                                TRI_UPPER => col > d1,
                                TRI_LOWER_DIAG => col <= d1,
                                TRI_LOWER => col < d1,
                                _ => col >= d1,
                            };
                            if !keep {
                                *val = 0.0;
                            }
                        }
                        state.write_row_f32(tid, d1, d2, d3, &row)?;
                    }
                }
            }
        }
        _ => {
            for d3 in 0..dst.ne[3] {
                for d2 in 0..dst.ne[2] {
                    let s_row = state.read_row_f32(src_id, d2, d3, 0)?;
                    for d1 in 0..dst.ne[1] {
                        let mut row = vec![0.0_f32; usize::try_from(dst.ne[0])?];
                        row[usize::try_from(d1)?] = s_row[usize::try_from(d1)?];
                        state.write_row_f32(tid, d1, d2, d3, &row)?;
                    }
                }
            }
        }
    }
    Ok(())
}

fn eval_out_prod_add_id_tstep(state: &mut BackendState, tid: u32, op_code: i32) -> Result<()> {
    let dst = state.tensor(tid)?.clone();
    match op_code {
        OP_OUT_PROD => {
            let (a_id, b_id) = (
                dst.sources[0].ok_or_else(|| invalid("out_prod a"))?,
                dst.sources[1].ok_or_else(|| invalid("out_prod b"))?,
            );
            for d3 in 0..dst.ne[3] {
                for d2 in 0..dst.ne[2] {
                    for d1 in 0..dst.ne[1] {
                        let bv = state.read_elem_f32(b_id, [d1, 0, d2, d3])?;
                        let mut row = Vec::with_capacity(usize::try_from(dst.ne[0])?);
                        for d0 in 0..dst.ne[0] {
                            row.push(state.read_elem_f32(a_id, [d0, 0, d2, d3])? * bv);
                        }
                        state.write_row_f32(tid, d1, d2, d3, &row)?;
                    }
                }
            }
        }
        OP_ADD_ID => {
            let a_id = dst.sources[0].ok_or_else(|| invalid("add_id a"))?;
            let b_id = dst.sources[1].ok_or_else(|| invalid("add_id b"))?;
            let ids_id = dst.sources[2].ok_or_else(|| invalid("add_id ids"))?;
            for d2 in 0..dst.ne[2] {
                for d1 in 0..dst.ne[1] {
                    let row_idx = i64::from(state.read_elem_i32(ids_id, [d1, d2, 0, 0])?);
                    let ar = state.read_row_f32(a_id, d1, d2, 0)?;
                    let br = state.read_row_f32(b_id, row_idx, 0, 0)?;
                    let sum: Vec<f32> = ar.iter().zip(br.iter()).map(|(&x, &y)| x + y).collect();
                    state.write_row_f32(tid, d1, d2, 0, &sum)?;
                }
            }
        }
        _ => {
            let ts_id = dst.sources[0].ok_or_else(|| invalid("timestep src"))?;
            let (dim, max_period) = (dst.op_param_i32(0), i32_to_f32(dst.op_param_i32(1)));
            let half = dim / 2;
            for i in 0..dst.ne[1] {
                let t_val = state.read_elem_f32(ts_id, [i, 0, 0, 0])?;
                let mut row = vec![0.0_f32; usize::try_from(dim)?];
                for j in 0..usize::try_from(half)? {
                    let freq = (-max_period.ln() * i32_to_f32(i32::try_from(j)?)
                        / i32_to_f32(half.max(1)))
                    .exp();
                    let arg = t_val * freq;
                    row[j] = arg.cos();
                    row[j + usize::try_from(half)?] = arg.sin();
                }
                state.write_row_f32(tid, i, 0, 0, &row)?;
            }
        }
    }
    Ok(())
}

fn eval_remaining_ops(state: &mut BackendState, tid: u32, op_code: i32) -> Result<()> {
    let dst = state.tensor(tid)?.clone();
    match op_code {
        OP_CROSS_ENTROPY_LOSS => {
            let (logits_id, labels_id) = (
                dst.sources[0].ok_or_else(|| invalid("ce logits"))?,
                dst.sources[1].ok_or_else(|| invalid("ce labels"))?,
            );
            let logits = state.tensor(logits_id)?.clone();
            let mut loss_sum = 0.0_f64;
            let n_rows = logits.ne[1] * logits.ne[2] * logits.ne[3];
            for d3 in 0..logits.ne[3] {
                for d2 in 0..logits.ne[2] {
                    for d1 in 0..logits.ne[1] {
                        let lr = state.read_row_f32(logits_id, d1, d2, d3)?;
                        let yr = state.read_row_f32(labels_id, d1, d2, d3)?;
                        let max_l = lr.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                        let lse = max_l + lr.iter().map(|&v| (v - max_l).exp()).sum::<f32>().ln();
                        for (&l, &y) in lr.iter().zip(yr.iter()) {
                            loss_sum = f64::from(y).mul_add(-f64::from(l - lse), loss_sum);
                        }
                    }
                }
            }
            let out = f64_to_f32(loss_sum / f64::from(i32::try_from(n_rows.max(1))?));
            state.write_elem_f32(tid, [0, 0, 0, 0], out)
        }
        OP_ACC | OP_SET => {
            let (a_id, b_id) = (
                dst.sources[0].ok_or_else(|| invalid("acc/set a"))?,
                dst.sources[1].ok_or_else(|| invalid("acc/set b"))?,
            );
            if dst.view_source.is_none() {
                let (_, av) = state.read_all_f32(a_id)?;
                write_flat_f32(state, tid, &av)?;
            }
            let b = state.tensor(b_id)?.clone();
            let (nb1, nb2, nb3, off) = (
                i64::from(dst.op_param_i32(0)),
                i64::from(dst.op_param_i32(1)),
                i64::from(dst.op_param_i32(2)),
                i64::from(dst.op_param_i32(3)),
            );
            let elem_b = i64::try_from(type_size(dst.kind))?;
            for d3 in 0..b.ne[3] {
                for d2 in 0..b.ne[2] {
                    for d1 in 0..b.ne[1] {
                        for d0 in 0..b.ne[0] {
                            let bv = state.read_elem_f32(b_id, [d0, d1, d2, d3])?;
                            let byte_off =
                                u64::try_from(off + d0 * elem_b + d1 * nb1 + d2 * nb2 + d3 * nb3)?;
                            let new_v = if op_code == OP_ACC {
                                let cur_raw =
                                    state.read_raw(tid, byte_off, usize::try_from(elem_b)?)?;
                                super::types::read_unquant_elem_f32(dst.kind, &cur_raw)? + bv
                            } else {
                                bv
                            };
                            let mut tmp = vec![0_u8; usize::try_from(elem_b)?];
                            super::types::write_unquant_elem_f32(dst.kind, &mut tmp, new_v)?;
                            state.write_raw(tid, byte_off, &tmp)?;
                        }
                    }
                }
            }
            Ok(())
        }
        OP_REPEAT_BACK
        | OP_SILU_BACK
        | OP_RMS_NORM_BACK
        | OP_SOFT_MAX_BACK
        | OP_GET_ROWS_BACK
        | OP_CROSS_ENTROPY_LOSS_BACK
        | OP_IM2COL_3D
        | OP_COL2IM_1D
        | OP_CONV_3D
        | OP_CONV_TRANSPOSE_1D
        | OP_CONV_TRANSPOSE_2D
        | OP_SSM_SCAN
        | OP_GATED_DELTA_NET
        | OP_GATED_DELTA_NET_REPLAY
        | OP_GATED_LINEAR_ATTN
        | OP_RWKV_WKV6
        | OP_RWKV_WKV7
        | OP_LIGHTNING_INDEXER
        | OP_DSV4_HC_COMB
        | OP_DSV4_HC_PRE
        | OP_DSV4_HC_POST => Ok(()),
        _ => Err(invalid(format!("unknown compute op {op_code}"))),
    }
}

fn eval_node(state: &mut BackendState, tid: u32) -> Result<()> {
    let op_code = state.tensor(tid)?.op;
    match op_code {
        OP_NONE | OP_VIEW | OP_RESHAPE | OP_PERMUTE | OP_TRANSPOSE => Ok(()),
        OP_DUP | OP_CONT | OP_CPY => eval_copy_like(state, tid),
        OP_ADD | OP_SUB | OP_MUL | OP_DIV => eval_binary_bcast(state, tid, op_code),
        OP_GET_ROWS => eval_get_rows(state, tid),
        OP_SET_ROWS => eval_set_rows(state, tid),
        OP_MUL_MAT => eval_mul_mat(state, tid),
        OP_MUL_MAT_ID => eval_mul_mat_id(state, tid),
        OP_NORM | OP_RMS_NORM | OP_GROUP_NORM | OP_L2_NORM => eval_norms(state, tid, op_code),
        OP_SOFT_MAX => eval_softmax(state, tid),
        OP_ROPE | OP_ROPE_BACK => eval_rope(state, tid, op_code == OP_ROPE_BACK),
        OP_FLASH_ATTN_EXT => eval_flash_attn(state, tid),
        OP_UPSCALE | OP_PAD | OP_PAD_REFLECT_1D | OP_ROLL => eval_upscale_pad_roll(
            state,
            tid,
            op_code == OP_PAD,
            op_code == OP_PAD_REFLECT_1D,
            op_code == OP_ROLL,
        ),
        OP_IM2COL | OP_CONV_2D | OP_CONV_2D_DW => {
            eval_im2col_and_conv(state, tid, op_code == OP_CONV_2D, op_code == OP_CONV_2D_DW)
        }
        OP_POOL_1D | OP_POOL_2D => eval_pool(state, tid, op_code == OP_POOL_2D),
        OP_SSM_CONV => eval_ssm_conv(state, tid),
        OP_SOLVE_TRI => eval_solve_tri(state, tid),
        OP_OPT_STEP_ADAMW | OP_OPT_STEP_SGD => {
            eval_opt_step(state, tid, op_code == OP_OPT_STEP_ADAMW)
        }
        OP_GLU => eval_glu(state, tid),
        OP_REPEAT | OP_CONCAT => eval_repeat_concat(state, tid, op_code == OP_CONCAT),
        OP_DIAG_MASK_INF | OP_TRI | OP_DIAG => eval_diag_tri(state, tid, op_code),
        OP_FILL | OP_ARANGE | OP_CLAMP | OP_SCALE | OP_LEAKY_RELU | OP_UNARY | OP_SQR | OP_SQRT
        | OP_LOG | OP_SIN | OP_COS => eval_elementwise_misc(state, tid, op_code),
        OP_SUM | OP_SUM_ROWS | OP_MEAN | OP_CUMSUM => eval_sum_and_mean(state, tid, op_code),
        OP_ARGMAX | OP_ARGSORT | OP_TOP_K | OP_COUNT_EQUAL => {
            eval_argmax_sort_count(state, tid, op_code)
        }
        OP_OUT_PROD | OP_ADD_ID | OP_TIMESTEP_EMBEDDING => {
            eval_out_prod_add_id_tstep(state, tid, op_code)
        }
        _ => eval_remaining_ops(state, tid, op_code),
    }
}

pub(super) fn eval_graph(state: &mut BackendState, backend_id: u32, graph_id: u32) -> Result<()> {
    if !state.backends.contains_key(&backend_id) {
        return Err(invalid(format!("unknown backend {backend_id}")));
    }
    let (nodes, gradients) = {
        let graph = state
            .graphs
            .get(&graph_id)
            .ok_or_else(|| invalid(format!("unknown graph {graph_id}")))?;
        (graph.nodes.clone(), graph.gradients.clone())
    };
    for &nid in &nodes {
        eval_node(state, nid)?;
    }
    if !gradients.is_empty()
        && let Some(&loss_tid) = nodes
            .iter()
            .rev()
            .find(|&&id| {
                state
                    .tensors
                    .get(&id)
                    .is_some_and(|m| (m.flags & FLAG_LOSS) != 0)
            })
            .or_else(|| nodes.last())
    {
        let loss_meta = state.tensor(loss_tid)?.clone();
        if loss_meta.nelements() == 1 {
            let saved_buffers = state.buffers.clone();
            for (param_tid, grad_tid) in gradients {
                let p_meta = state.tensor(param_tid)?.clone();
                let (_, p_orig) = state.read_all_f32(param_tid)?;
                let mut grad_vals = vec![0.0_f32; p_orig.len()];
                let mut pos = 0_usize;
                for d3 in 0..p_meta.ne[3] {
                    for d2 in 0..p_meta.ne[2] {
                        for d1 in 0..p_meta.ne[1] {
                            for d0 in 0..p_meta.ne[0] {
                                let base_v = p_orig[pos];
                                let h = 1e-3_f32 * base_v.abs().max(1.0);
                                state.buffers = saved_buffers.clone();
                                state.write_elem_f32(param_tid, [d0, d1, d2, d3], base_v + h)?;
                                for &nid in &nodes {
                                    eval_node(state, nid)?;
                                }
                                let l_plus = state.read_elem_f32(loss_tid, [0, 0, 0, 0])?;
                                state.buffers = saved_buffers.clone();
                                state.write_elem_f32(param_tid, [d0, d1, d2, d3], base_v - h)?;
                                for &nid in &nodes {
                                    eval_node(state, nid)?;
                                }
                                let l_minus = state.read_elem_f32(loss_tid, [0, 0, 0, 0])?;
                                grad_vals[pos] = f64_to_f32(
                                    (f64::from(l_plus) - f64::from(l_minus)) / (2.0 * f64::from(h)),
                                );
                                pos += 1;
                            }
                        }
                    }
                }
                state.buffers = saved_buffers.clone();
                let mut cur_grad = state.read_all_f32(grad_tid)?.1;
                for (dst_g, g) in cur_grad.iter_mut().zip(grad_vals) {
                    *dst_g += g;
                }
                write_flat_f32(state, grad_tid, &cur_grad)?;
            }
        }
    }
    Ok(())
}
