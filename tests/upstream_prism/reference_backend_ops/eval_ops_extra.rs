//! Extended clean-room GGML tensor op evaluators (attention, `RoPE`, vision, recurrent, optimizer).
//!
//! No `bonsai_prism_runtime`, `bonsai_gguf`, or native GGML libraries are linked or called.

use std::f32::consts::SQRT_2;

use super::state::BackendState;
use super::types::{
    ROPE_TYPE_IMROPE, ROPE_TYPE_MROPE, ROPE_TYPE_NEOX, ROPE_TYPE_VISION, SCALE_FLAG_ALIGN_CORNERS,
    SCALE_MODE_BILINEAR, TensorMeta, f32_to_i64_trunc, f64_to_f32, i32_to_f32, usize_to_f32,
};
use crate::{Result, invalid};

type RowWrite = (i64, i64, i64, Vec<f32>);

pub(super) fn gelu_erf(x: f32) -> f32 {
    0.5 * x * (1.0 + libm_erf_approx(x / SQRT_2))
}

fn libm_erf_approx(x: f32) -> f32 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let t = 1.0 / 0.327_591_1_f32.mul_add(x.abs(), 1.0);
    let poly = t * t.mul_add(
        t.mul_add(
            t.mul_add(t.mul_add(1.061_405_4, -1.453_152), 1.421_413_7),
            -0.284_496_74,
        ),
        0.254_829_6,
    );
    sign * poly.mul_add(-(-x * x).exp(), 1.0)
}

fn rope_yarn_corr(n_dims: f32, n_ctx_orig: f32, n_rot: f32, base: f32) -> f32 {
    n_dims * (n_ctx_orig / (n_rot * 2.0 * std::f32::consts::PI)).ln() / (2.0 * base.ln())
}

fn yarn_ramp(low: f32, high: f32, idx: f32) -> f32 {
    let span = if (low - high).abs() < 1e-6 {
        0.001
    } else {
        high - low
    };
    1.0 - ((idx - low) / span).clamp(0.0, 1.0)
}

struct RopeCfg {
    n_dims: usize,
    mode: i32,
    theta_scale: f32,
    mscale: f32,
    freq_scale: f32,
    ext_factor: f32,
    low: f32,
    high: f32,
    sections: [i32; 4],
    n_offs: i32,
    is_back: bool,
    freq_factors: Option<u32>,
}

fn rope_cfg_from(dst: &TensorMeta, is_back: bool) -> Result<RopeCfg> {
    let n_dims = usize::try_from(dst.op_param_i32(1).max(0))?;
    let freq_base = dst.op_param_f32(5);
    let freq_scale = dst.op_param_f32(6);
    let ext_factor = dst.op_param_f32(7);
    let attn_factor = dst.op_param_f32(8);
    let n_ctx_orig = i32_to_f32(dst.op_param_i32(4));
    let lo = rope_yarn_corr(
        usize_to_f32(n_dims),
        n_ctx_orig,
        dst.op_param_f32(9),
        freq_base,
    );
    let hi = rope_yarn_corr(
        usize_to_f32(n_dims),
        n_ctx_orig,
        dst.op_param_f32(10),
        freq_base,
    );
    let mscale = if ext_factor == 0.0 {
        attn_factor
    } else {
        attn_factor * (0.1_f32.mul_add((1.0_f32 / freq_scale).ln(), 1.0))
    };
    Ok(RopeCfg {
        n_dims,
        mode: dst.op_param_i32(2),
        theta_scale: freq_base.powf(-2.0 / usize_to_f32(n_dims.max(2))),
        mscale,
        freq_scale,
        ext_factor,
        low: lo.min(hi),
        high: lo.max(hi),
        sections: [
            dst.op_param_i32(11),
            dst.op_param_i32(12),
            dst.op_param_i32(13),
            dst.op_param_i32(14),
        ],
        n_offs: dst.op_param_i32(15),
        is_back,
        freq_factors: dst.sources[2],
    })
}

fn eval_rope_row(
    state: &BackendState,
    cfg: &RopeCfg,
    row: &[f32],
    pos_id: u32,
    i2: i64,
    src_ne2: i64,
) -> Result<Vec<f32>> {
    let mrope_enabled = (cfg.mode & ROPE_TYPE_MROPE) == ROPE_TYPE_MROPE;
    let interleaved_mrope = cfg.mode == ROPE_TYPE_IMROPE;
    let is_neox = (cfg.mode & ROPE_TYPE_NEOX) != 0;
    let is_vision = cfg.mode == ROPE_TYPE_VISION;
    let base_pos = state.read_elem_i32(pos_id, [i2, 0, 0, 0])? + cfg.n_offs;
    let mut out_row = row.to_vec();
    let mut pair = 0_usize;
    while pair < cfg.n_dims {
        let ic = pair / 2;
        let ff = if let Some(ff_id) = cfg.freq_factors {
            state.read_elem_f32(ff_id, [i64::try_from(ic)?, 0, 0, 0])?
        } else {
            1.0
        };
        let pos_val = if mrope_enabled {
            let sec_idx = if interleaved_mrope {
                ic % 3
            } else if ic * 2 < usize::try_from(cfg.sections[0].max(0))? {
                0
            } else if ic * 2 < usize::try_from((cfg.sections[0] + cfg.sections[1]).max(0))? {
                1
            } else {
                2
            };
            let p_idx = i64::try_from(sec_idx)? * src_ne2 + i2;
            state.read_elem_i32(pos_id, [p_idx, 0, 0, 0])? + cfg.n_offs
        } else {
            base_pos
        };
        let inv_extrap = cfg.theta_scale.powf(usize_to_f32(ic)) / ff;
        let inv_interp = inv_extrap * cfg.freq_scale;
        let theta = i32_to_f32(pos_val)
            * if cfg.ext_factor == 0.0 {
                inv_interp
            } else {
                let ramp = yarn_ramp(cfg.low, cfg.high, usize_to_f32(ic));
                let blend = ramp * cfg.ext_factor;
                inv_interp.mul_add(1.0 - blend, inv_extrap * blend)
            };
        let cos_t = theta.cos() * cfg.mscale;
        let sin_t = if cfg.is_back {
            -theta.sin()
        } else {
            theta.sin()
        } * cfg.mscale;
        let (idx0, idx1) = if is_neox || is_vision {
            (ic, ic + cfg.n_dims / 2)
        } else {
            (pair, pair + 1)
        };
        let (x0, x1) = (row[idx0], row[idx1]);
        out_row[idx0] = x0.mul_add(cos_t, -(x1 * sin_t));
        out_row[idx1] = x0.mul_add(sin_t, x1 * cos_t);
        pair += 2;
    }
    Ok(out_row)
}

pub(super) fn eval_rope(state: &mut BackendState, tid: u32, is_back: bool) -> Result<()> {
    let dst = state.tensor(tid)?.clone();
    let src_id = dst.sources[0].ok_or_else(|| invalid("rope missing src"))?;
    let pos_id = dst.sources[1].ok_or_else(|| invalid("rope missing pos"))?;
    let cfg = rope_cfg_from(&dst, is_back)?;
    let src_meta = state.tensor(src_id)?.clone();
    let mut rows = Vec::new();
    for i3 in 0..src_meta.ne[3] {
        for i2 in 0..src_meta.ne[2] {
            for i1 in 0..src_meta.ne[1] {
                let row = state.read_row_f32(src_id, i1, i2, i3)?;
                let out_row = eval_rope_row(state, &cfg, &row, pos_id, i2, src_meta.ne[2])?;
                rows.push((i1, i2, i3, out_row));
            }
        }
    }
    for (i1, i2, i3, out_row) in rows {
        state.write_row_f32(tid, i1, i2, i3, &out_row)?;
    }
    Ok(())
}

pub(super) fn eval_flash_attn(state: &mut BackendState, tid: u32) -> Result<()> {
    let dst = state.tensor(tid)?.clone();
    let q_id = dst.sources[0].ok_or_else(|| invalid("flash_attn missing q"))?;
    let k_id = dst.sources[1].ok_or_else(|| invalid("flash_attn missing k"))?;
    let v_id = dst.sources[2].ok_or_else(|| invalid("flash_attn missing v"))?;
    let mask_id = dst.sources[3];
    let sinks_id = dst.sources[4];
    let (scale, max_bias, softcap) = (
        dst.op_param_f32(0),
        dst.op_param_f32(1),
        dst.op_param_f32(2),
    );
    let q_meta = state.tensor(q_id)?.clone();
    let k_meta = state.tensor(k_id)?.clone();
    let v_meta = state.tensor(v_id)?.clone();
    let n_heads = q_meta.ne[2];
    let gqa_ratio = (n_heads / k_meta.ne[2].max(1)).max(1);
    let dv = usize::try_from(v_meta.ne[0])?;

    let mut writes = Vec::new();
    for batch in 0..q_meta.ne[3] {
        for head in 0..n_heads {
            let kv_head = head / gqa_ratio;
            let slope = alibi_slope(head, n_heads, max_bias);
            let sink_logit = sinks_id
                .map(|sid| state.read_elem_f32(sid, [head, 0, 0, 0]))
                .transpose()?;
            for tok in 0..q_meta.ne[1] {
                let q_row = state.read_row_f32(q_id, tok, head, batch)?;
                let mut logits = Vec::with_capacity(usize::try_from(k_meta.ne[1])?);
                let mut max_logit = sink_logit.unwrap_or(f32::NEG_INFINITY);
                for kv_idx in 0..k_meta.ne[1] {
                    let k_row = state.read_row_f32(k_id, kv_idx, kv_head, batch % k_meta.ne[3])?;
                    let dot: f64 = q_row.iter().zip(k_row.iter()).fold(0.0, |acc, (&qv, &kv)| {
                        f64::from(qv).mul_add(f64::from(kv), acc)
                    });
                    let mut val = f64_to_f32(dot) * scale;
                    if softcap > 0.0 {
                        val = softcap * (val / softcap).tanh();
                    }
                    if let Some(mid) = mask_id {
                        let mm = state.tensor(mid)?;
                        let mv = state.read_elem_f32(
                            mid,
                            [kv_idx, tok % mm.ne[1], head % mm.ne[2], batch % mm.ne[3]],
                        )?;
                        val = slope.mul_add(mv, val);
                    }
                    max_logit = max_logit.max(val);
                    logits.push(val);
                }
                let mut sum_exp = sink_logit.map_or(0.0, |sl| (sl - max_logit).exp());
                for logit in &mut logits {
                    *logit = (*logit - max_logit).exp();
                    sum_exp += *logit;
                }
                let mut acc = vec![0.0_f64; dv];
                for (kv_idx, &prob) in logits.iter().enumerate() {
                    let weight = if sum_exp > 0.0 { prob / sum_exp } else { 0.0 };
                    let v_row = state.read_row_f32(
                        v_id,
                        i64::try_from(kv_idx)?,
                        kv_head,
                        batch % v_meta.ne[3],
                    )?;
                    for (slot, &vv) in acc.iter_mut().zip(v_row.iter()) {
                        *slot = f64::from(weight).mul_add(f64::from(vv), *slot);
                    }
                }
                let out_row: Vec<f32> = acc.into_iter().map(f64_to_f32).collect();
                writes.push((head, tok, batch, out_row));
            }
        }
    }
    for (head, tok, batch, row) in writes {
        state.write_row_f32(tid, head, tok, batch, &row)?;
    }
    Ok(())
}

fn alibi_slope(head: i64, n_heads: i64, max_bias: f32) -> f32 {
    if max_bias <= 0.0 || n_heads <= 0 {
        return 1.0;
    }
    let n_log2 = u32::try_from(n_heads).ok().map_or(0, u32::ilog2);
    let base_pow = (-8.0 / i32_to_f32(i32::try_from(n_log2).unwrap_or(0)).exp2()).exp2();
    let h_val = i32::try_from(head + 1).unwrap_or(1);
    base_pow.powi(h_val)
}

fn eval_roll(
    state: &BackendState,
    dst: &TensorMeta,
    src_id: u32,
    src: &TensorMeta,
) -> Result<Vec<RowWrite>> {
    let shifts = [
        dst.op_param_i32(0),
        dst.op_param_i32(1),
        dst.op_param_i32(2),
        dst.op_param_i32(3),
    ];
    let mut vals = Vec::new();
    for i3 in 0..dst.ne[3] {
        for i2 in 0..dst.ne[2] {
            for i1 in 0..dst.ne[1] {
                let mut row = Vec::with_capacity(usize::try_from(dst.ne[0])?);
                let s1 = (i1 - i64::from(shifts[1])).rem_euclid(src.ne[1]);
                let s2 = (i2 - i64::from(shifts[2])).rem_euclid(src.ne[2]);
                let s3 = (i3 - i64::from(shifts[3])).rem_euclid(src.ne[3]);
                for i0 in 0..dst.ne[0] {
                    let s0 = (i0 - i64::from(shifts[0])).rem_euclid(src.ne[0]);
                    row.push(state.read_elem_f32(src_id, [s0, s1, s2, s3])?);
                }
                vals.push((i1, i2, i3, row));
            }
        }
    }
    Ok(vals)
}

fn eval_pad_reflect(
    state: &BackendState,
    dst: &TensorMeta,
    src_id: u32,
    src: &TensorMeta,
) -> Result<Vec<RowWrite>> {
    let lp0 = i64::from(dst.op_param_i32(0));
    let mut vals = Vec::new();
    for i3 in 0..dst.ne[3] {
        for i2 in 0..dst.ne[2] {
            for i1 in 0..dst.ne[1] {
                let src_row = state.read_row_f32(src_id, i1, i2, i3)?;
                let mut row = Vec::with_capacity(usize::try_from(dst.ne[0])?);
                for i0 in 0..dst.ne[0] {
                    let mut rel = i0 - lp0;
                    if rel < 0 {
                        rel = -rel;
                    } else if rel >= src.ne[0] {
                        rel = 2 * src.ne[0] - 2 - rel;
                    }
                    let clamped = usize::try_from(rel.clamp(0, src.ne[0] - 1))?;
                    row.push(src_row[clamped]);
                }
                vals.push((i1, i2, i3, row));
            }
        }
    }
    Ok(vals)
}

fn eval_pad_general(
    state: &BackendState,
    dst: &TensorMeta,
    src_id: u32,
    src: &TensorMeta,
) -> Result<Vec<RowWrite>> {
    let lp = [
        i64::from(dst.op_param_i32(0)),
        i64::from(dst.op_param_i32(2)),
        i64::from(dst.op_param_i32(4)),
        i64::from(dst.op_param_i32(6)),
    ];
    let circular = dst.op_param_i32(8) != 0;
    let mut vals = Vec::new();
    for i3 in 0..dst.ne[3] {
        for i2 in 0..dst.ne[2] {
            for i1 in 0..dst.ne[1] {
                let mut row = Vec::with_capacity(usize::try_from(dst.ne[0])?);
                for i0 in 0..dst.ne[0] {
                    let raw_idx = [i0 - lp[0], i1 - lp[1], i2 - lp[2], i3 - lp[3]];
                    let val = if circular {
                        state.read_elem_f32(
                            src_id,
                            [
                                raw_idx[0].rem_euclid(src.ne[0]),
                                raw_idx[1].rem_euclid(src.ne[1]),
                                raw_idx[2].rem_euclid(src.ne[2]),
                                raw_idx[3].rem_euclid(src.ne[3]),
                            ],
                        )?
                    } else if (0..4).all(|d| raw_idx[d] >= 0 && raw_idx[d] < src.ne[d]) {
                        state.read_elem_f32(src_id, raw_idx)?
                    } else {
                        0.0
                    };
                    row.push(val);
                }
                vals.push((i1, i2, i3, row));
            }
        }
    }
    Ok(vals)
}

fn eval_upscale(
    state: &BackendState,
    dst: &TensorMeta,
    src_id: u32,
    src: &TensorMeta,
) -> Result<Vec<RowWrite>> {
    let mode_bits = dst.op_param_i32(0);
    let base_mode = mode_bits & 0xff;
    let align = (mode_bits & SCALE_FLAG_ALIGN_CORNERS) != 0;
    let scale_coord = |out_idx: i64, out_len: i64, in_len: i64| -> f32 {
        if align && out_len > 1 && in_len > 1 {
            i32_to_f32(i32::try_from(out_idx).unwrap_or(0))
                * i32_to_f32(i32::try_from(in_len - 1).unwrap_or(1))
                / i32_to_f32(i32::try_from(out_len - 1).unwrap_or(1))
        } else {
            let step = i32_to_f32(i32::try_from(in_len).unwrap_or(1))
                / i32_to_f32(i32::try_from(out_len).unwrap_or(1));
            (i32_to_f32(i32::try_from(out_idx).unwrap_or(0)) + 0.5).mul_add(step, -0.5)
        }
    };
    let mut vals = Vec::new();
    for i3 in 0..dst.ne[3] {
        for i2 in 0..dst.ne[2] {
            for i1 in 0..dst.ne[1] {
                let mut row = Vec::with_capacity(usize::try_from(dst.ne[0])?);
                for i0 in 0..dst.ne[0] {
                    let val = if base_mode == SCALE_MODE_BILINEAR {
                        let sx = scale_coord(i0, dst.ne[0], src.ne[0])
                            .clamp(0.0, i32_to_f32(i32::try_from(src.ne[0] - 1).unwrap_or(0)));
                        let sy = scale_coord(i1, dst.ne[1], src.ne[1])
                            .clamp(0.0, i32_to_f32(i32::try_from(src.ne[1] - 1).unwrap_or(0)));
                        let x0 = f32_to_i64_trunc(sx.floor()).clamp(0, src.ne[0] - 1);
                        let y0 = f32_to_i64_trunc(sy.floor()).clamp(0, src.ne[1] - 1);
                        let (x1, y1) = ((x0 + 1).min(src.ne[0] - 1), (y0 + 1).min(src.ne[1] - 1));
                        let (fx, fy) = (sx - sx.floor(), sy - sy.floor());
                        let v00 = state.read_elem_f32(src_id, [x0, y0, i2, i3])?;
                        let v10 = state.read_elem_f32(src_id, [x1, y0, i2, i3])?;
                        let v01 = state.read_elem_f32(src_id, [x0, y1, i2, i3])?;
                        let v11 = state.read_elem_f32(src_id, [x1, y1, i2, i3])?;
                        let top = v00.mul_add(1.0 - fx, v10 * fx);
                        let bot = v01.mul_add(1.0 - fx, v11 * fx);
                        top.mul_add(1.0 - fy, bot * fy)
                    } else {
                        let sx = (i0 * src.ne[0] / dst.ne[0]).clamp(0, src.ne[0] - 1);
                        let sy = (i1 * src.ne[1] / dst.ne[1]).clamp(0, src.ne[1] - 1);
                        state.read_elem_f32(src_id, [sx, sy, i2, i3])?
                    };
                    row.push(val);
                }
                vals.push((i1, i2, i3, row));
            }
        }
    }
    Ok(vals)
}

pub(super) fn eval_upscale_pad_roll(
    state: &mut BackendState,
    tid: u32,
    is_pad: bool,
    is_reflect: bool,
    is_roll: bool,
) -> Result<()> {
    let dst = state.tensor(tid)?.clone();
    let src_id = dst.sources[0].ok_or_else(|| invalid("spatial op missing src"))?;
    let src = state.tensor(src_id)?.clone();
    let vals = if is_roll {
        eval_roll(state, &dst, src_id, &src)?
    } else if is_reflect {
        eval_pad_reflect(state, &dst, src_id, &src)?
    } else if is_pad {
        eval_pad_general(state, &dst, src_id, &src)?
    } else {
        eval_upscale(state, &dst, src_id, &src)?
    };
    for (i1, i2, i3, row) in vals {
        state.write_row_f32(tid, i1, i2, i3, &row)?;
    }
    Ok(())
}

pub(super) fn eval_im2col_and_conv(
    state: &mut BackendState,
    tid: u32,
    is_conv2d: bool,
    is_dw: bool,
) -> Result<()> {
    let dst = state.tensor(tid)?.clone();
    let kern_id = dst.sources[0].ok_or_else(|| invalid("conv/im2col missing kernel"))?;
    let inp_id = dst.sources[1].ok_or_else(|| invalid("conv/im2col missing input"))?;
    let kern = state.tensor(kern_id)?.clone();
    let inp = state.tensor(inp_id)?.clone();
    let (s0, s1) = (
        i64::from(dst.op_param_i32(0).max(1)),
        i64::from(dst.op_param_i32(1).max(1)),
    );
    let (p0, p1) = (
        i64::from(dst.op_param_i32(2)),
        i64::from(dst.op_param_i32(3)),
    );
    let (d0, d1) = (
        i64::from(dst.op_param_i32(4).max(1)),
        i64::from(dst.op_param_i32(5).max(1)),
    );

    if !is_conv2d && !is_dw {
        let is_2d = dst.op_param_i32(6) != 0;
        let (kw, kh, ic) = (
            kern.ne[0],
            if is_2d { kern.ne[1] } else { 1 },
            if is_2d { kern.ne[2] } else { kern.ne[1] },
        );
        let mut writes = Vec::new();
        for batch in 0..dst.ne[3] {
            for oy in 0..dst.ne[2] {
                for ox in 0..dst.ne[1] {
                    let mut col = Vec::with_capacity(usize::try_from(dst.ne[0])?);
                    for c in 0..ic {
                        for ky in 0..kh {
                            for kx in 0..kw {
                                let ix = ox * s0 - p0 + kx * d0;
                                let v = if is_2d {
                                    let iy = oy * s1 - p1 + ky * d1;
                                    if ix >= 0 && ix < inp.ne[0] && iy >= 0 && iy < inp.ne[1] {
                                        state.read_elem_f32(inp_id, [ix, iy, c, batch])?
                                    } else {
                                        0.0
                                    }
                                } else if ix >= 0 && ix < inp.ne[0] {
                                    state.read_elem_f32(inp_id, [ix, c, oy, batch])?
                                } else {
                                    0.0
                                };
                                col.push(v);
                            }
                        }
                    }
                    writes.push((ox, oy, batch, col));
                }
            }
        }
        for (ox, oy, batch, col) in writes {
            state.write_row_f32(tid, ox, oy, batch, &col)?;
        }
        return Ok(());
    }

    let mut writes = Vec::new();
    for batch in 0..dst.ne[3] {
        for oc in 0..dst.ne[2] {
            for oy in 0..dst.ne[1] {
                let mut row = Vec::with_capacity(usize::try_from(dst.ne[0])?);
                for ox in 0..dst.ne[0] {
                    let mut sum = 0.0_f64;
                    let c_range = if is_dw { oc..oc + 1 } else { 0..kern.ne[2] };
                    for ic in c_range {
                        for ky in 0..kern.ne[1] {
                            for kx in 0..kern.ne[0] {
                                let ix = ox * s0 - p0 + kx * d0;
                                let iy = oy * s1 - p1 + ky * d1;
                                if ix >= 0 && ix < inp.ne[0] && iy >= 0 && iy < inp.ne[1] {
                                    let xv = state.read_elem_f32(inp_id, [ix, iy, ic, batch])?;
                                    let wv = if is_dw {
                                        state.read_elem_f32(kern_id, [kx, ky, 0, oc])?
                                    } else {
                                        state.read_elem_f32(kern_id, [kx, ky, ic, oc])?
                                    };
                                    sum = f64::from(xv).mul_add(f64::from(wv), sum);
                                }
                            }
                        }
                    }
                    row.push(f64_to_f32(sum));
                }
                writes.push((oy, oc, batch, row));
            }
        }
    }
    for (oy, oc, batch, row) in writes {
        state.write_row_f32(tid, oy, oc, batch, &row)?;
    }
    Ok(())
}

pub(super) fn eval_pool(state: &mut BackendState, tid: u32, is_2d: bool) -> Result<()> {
    let dst = state.tensor(tid)?.clone();
    let src_id = dst.sources[0].ok_or_else(|| invalid("pool missing src"))?;
    let src = state.tensor(src_id)?.clone();
    let is_max = dst.op_param_i32(0) == 0;
    let k0 = i64::from(dst.op_param_i32(1));
    let (k1, s0, s1, p0, p1) = if is_2d {
        (
            i64::from(dst.op_param_i32(2)),
            i64::from(dst.op_param_i32(3).max(1)),
            i64::from(dst.op_param_i32(4).max(1)),
            f32_to_i64_trunc(dst.op_param_f32(5)),
            f32_to_i64_trunc(dst.op_param_f32(6)),
        )
    } else {
        (
            1,
            i64::from(dst.op_param_i32(2).max(1)),
            1,
            i64::from(dst.op_param_i32(3)),
            0,
        )
    };
    let mut writes = Vec::new();
    for i3 in 0..dst.ne[3] {
        for i2 in 0..dst.ne[2] {
            for i1 in 0..dst.ne[1] {
                let mut row = Vec::with_capacity(usize::try_from(dst.ne[0])?);
                for i0 in 0..dst.ne[0] {
                    let mut best = f32::NEG_INFINITY;
                    let mut sum = 0.0_f64;
                    let ky_max = if is_2d { k1 } else { 1 };
                    for ky in 0..ky_max {
                        for kx in 0..k0 {
                            let ix = i0 * s0 - p0 + kx;
                            let iy = if is_2d { i1 * s1 - p1 + ky } else { i1 };
                            if ix >= 0 && ix < src.ne[0] && iy >= 0 && iy < src.ne[1] {
                                let v = state.read_elem_f32(src_id, [ix, iy, i2, i3])?;
                                best = best.max(v);
                                sum += f64::from(v);
                            }
                        }
                    }
                    let denom = usize_to_f32(usize::try_from(k0 * ky_max).unwrap_or(1));
                    row.push(if is_max {
                        best
                    } else {
                        f64_to_f32(sum) / denom
                    });
                }
                writes.push((i1, i2, i3, row));
            }
        }
    }
    for (i1, i2, i3, row) in writes {
        state.write_row_f32(tid, i1, i2, i3, &row)?;
    }
    Ok(())
}

pub(super) fn eval_ssm_conv(state: &mut BackendState, tid: u32) -> Result<()> {
    let dst = state.tensor(tid)?.clone();
    let sx_id = dst.sources[0].ok_or_else(|| invalid("ssm_conv missing sx"))?;
    let c_id = dst.sources[1].ok_or_else(|| invalid("ssm_conv missing c"))?;
    let d_conv = state.tensor(c_id)?.ne[0];
    let mut writes = Vec::new();
    for i3 in 0..dst.ne[3] {
        for i2 in 0..dst.ne[2] {
            for i1 in 0..dst.ne[1] {
                let mut row = Vec::with_capacity(usize::try_from(dst.ne[0])?);
                for i0 in 0..dst.ne[0] {
                    let mut acc = 0.0_f64;
                    for k in 0..d_conv {
                        let sv = state.read_elem_f32(sx_id, [i0, i1 + k, i2, i3])?;
                        let cw = state.read_elem_f32(c_id, [k, i0, 0, 0])?;
                        acc = f64::from(sv).mul_add(f64::from(cw), acc);
                    }
                    row.push(f64_to_f32(acc));
                }
                writes.push((i1, i2, i3, row));
            }
        }
    }
    for (i1, i2, i3, row) in writes {
        state.write_row_f32(tid, i1, i2, i3, &row)?;
    }
    Ok(())
}

pub(super) fn eval_opt_step(state: &mut BackendState, tid: u32, is_adamw: bool) -> Result<()> {
    let dst = state.tensor(tid)?;
    let par_id = dst.sources[0].ok_or_else(|| invalid("opt_step missing param"))?;
    let grad_id = dst.sources[1].ok_or_else(|| invalid("opt_step missing grad"))?;
    let (_, mut par) = state.read_all_f32(par_id)?;
    let (_, grad) = state.read_all_f32(grad_id)?;
    if is_adamw {
        let m_id = dst.sources[2].ok_or_else(|| invalid("adamw missing m"))?;
        let v_id = dst.sources[3].ok_or_else(|| invalid("adamw missing v"))?;
        let params_id = dst.sources[4].ok_or_else(|| invalid("adamw missing params"))?;
        let (_, mut m) = state.read_all_f32(m_id)?;
        let (_, mut v) = state.read_all_f32(v_id)?;
        let (_, p) = state.read_all_f32(params_id)?;
        let [alpha, beta1, beta2, eps, wd, beta1h, beta2h, ..] = p[..] else {
            return Err(invalid("adamw params too short"));
        };
        for idx in 0..par.len() {
            m[idx] = beta1.mul_add(m[idx], (1.0 - beta1) * grad[idx]);
            v[idx] = beta2.mul_add(v[idx], (1.0 - beta2) * grad[idx] * grad[idx]);
            let mh = m[idx] * beta1h;
            let vh = v[idx] * beta2h;
            par[idx] = par[idx].mul_add(alpha.mul_add(-wd, 1.0), -(alpha * mh / (vh.sqrt() + eps)));
        }
        write_flat_f32(state, m_id, &m)?;
        write_flat_f32(state, v_id, &v)?;
    } else {
        let params_id = dst.sources[2].ok_or_else(|| invalid("sgd missing params"))?;
        let (_, p) = state.read_all_f32(params_id)?;
        let (lr, wd) = (p[0], p[1]);
        for (pv, &gv) in par.iter_mut().zip(grad.iter()) {
            *pv = (*pv).mul_add(wd.mul_add(-lr, 1.0), -(lr * gv));
        }
    }
    write_flat_f32(state, par_id, &par)
}

pub(super) fn write_flat_f32(state: &mut BackendState, tid: u32, vals: &[f32]) -> Result<()> {
    let meta = state.tensor(tid)?.clone();
    let ne0 = usize::try_from(meta.ne[0])?;
    let mut off = 0_usize;
    for i3 in 0..meta.ne[3] {
        for i2 in 0..meta.ne[2] {
            for i1 in 0..meta.ne[1] {
                state.write_row_f32(tid, i1, i2, i3, &vals[off..off + ne0])?;
                off += ne0;
            }
        }
    }
    Ok(())
}

fn solve_tri_left(
    mat_a: &[Vec<f64>],
    mat_x: &mut [Vec<f64>],
    n: usize,
    k_cols: usize,
    lower: bool,
    uni: bool,
) {
    for col in 0..k_cols {
        let order: Vec<usize> = if lower {
            (0..n).collect()
        } else {
            (0..n).rev().collect()
        };
        for i in order {
            let mut sum = mat_x[i][col];
            let range = if lower { 0..i } else { i + 1..n };
            for j in range {
                sum = mat_a[i][j].mul_add(-mat_x[j][col], sum);
            }
            if !uni && mat_a[i][i] != 0.0 {
                sum /= mat_a[i][i];
            }
            mat_x[i][col] = sum;
        }
    }
}

fn solve_tri_right(
    mat_a: &[Vec<f64>],
    mat_x: &mut [Vec<f64>],
    k_cols: usize,
    lower: bool,
    uni: bool,
) {
    for x_row in mat_x {
        let order: Vec<usize> = if lower {
            (0..k_cols).rev().collect()
        } else {
            (0..k_cols).collect()
        };
        for j in order {
            let mut sum = x_row[j];
            let range = if lower { j + 1..k_cols } else { 0..j };
            for c in range {
                sum = x_row[c].mul_add(-mat_a[c][j], sum);
            }
            if !uni && mat_a[j][j] != 0.0 {
                sum /= mat_a[j][j];
            }
            x_row[j] = sum;
        }
    }
}

pub(super) fn eval_solve_tri(state: &mut BackendState, tid: u32) -> Result<()> {
    let dst = state.tensor(tid)?.clone();
    let a_id = dst.sources[0].ok_or_else(|| invalid("solve_tri missing A"))?;
    let b_id = dst.sources[1].ok_or_else(|| invalid("solve_tri missing B"))?;
    let (left, lower, uni) = (
        dst.op_param_i32(0) != 0,
        dst.op_param_i32(1) != 0,
        dst.op_param_i32(2) != 0,
    );
    let n = usize::try_from(dst.ne[1])?;
    let k_cols = usize::try_from(dst.ne[0])?;
    for i3 in 0..dst.ne[3] {
        for i2 in 0..dst.ne[2] {
            let mut mat_x = vec![vec![0.0_f64; k_cols]; n];
            for (row_idx, x_row) in mat_x.iter_mut().enumerate() {
                let r_row = state.read_row_f32(b_id, i64::try_from(row_idx)?, i2, i3)?;
                for (dst_c, &v) in x_row.iter_mut().zip(r_row.iter()) {
                    *dst_c = f64::from(v);
                }
            }
            let a_dim = if left { n } else { k_cols };
            let mut mat_a = vec![vec![0.0_f64; a_dim]; a_dim];
            for (row_idx, a_row_out) in mat_a.iter_mut().enumerate() {
                let a_row = state.read_row_f32(a_id, i64::try_from(row_idx)?, i2, i3)?;
                for (dst_c, &v) in a_row_out.iter_mut().zip(a_row.iter()) {
                    *dst_c = f64::from(v);
                }
            }
            if left {
                solve_tri_left(&mat_a, &mut mat_x, n, k_cols, lower, uni);
            } else {
                solve_tri_right(&mat_a, &mut mat_x, k_cols, lower, uni);
            }
            for (row_idx, r_vals) in mat_x.into_iter().enumerate() {
                let f_row: Vec<f32> = r_vals.into_iter().map(f64_to_f32).collect();
                state.write_row_f32(tid, i64::try_from(row_idx)?, i2, i3, &f_row)?;
            }
        }
    }
    Ok(())
}
