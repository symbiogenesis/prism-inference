//! Opt-in bounded triangular GDN chunks; no cumulative-decay division.

use super::{TileReuse, TiledPrefill, project_tile, project_tile_pair, reborrow_executor};
use crate::graph::{
    Arithmetic, Geometry, LayerDelta, LayerState, RecurrentAttention, Transforms, copy_into,
    finite, l2_norm, resize_into, sigmoid, small_float,
};
use crate::kernels::pinned_gdn::ChunkInput as Input;
use crate::kernels::{add_scaled_slice_in_place, gdn_wy_update_four_rows, scale_slice_into};
use crate::model::lora::{RankOneAdapter, Scratch as AdapterScratch};
use crate::{Result, invalid};

#[cfg(test)]
fn head(
    arithmetic: Arithmetic,
    previous: &[f32],
    inputs: &[Input<'_>],
    matrix: &mut [f32],
    output: &mut Vec<f32>,
) -> Result<()> {
    let Some(first) = inputs.first() else {
        return Err(invalid("empty GDN chunk"));
    };
    let width = first.key.len();
    let count = inputs.len();
    resize_into(
        count
            .checked_mul(width)
            .ok_or_else(|| invalid("GDN chunk output overflow"))?,
        output,
    )?;
    head_into(arithmetic, previous, inputs, None, matrix, output)
}

const KEY_DOT_STRIDE: usize = 2 * 32 * 32;

fn fill_key_head_dots(
    arithmetic: Arithmetic,
    mixed: &[Vec<f32>],
    key_width: usize,
    width: usize,
    key_head: usize,
    dots: &mut [f32],
) -> Result<()> {
    let q_range = key_head * width..(key_head + 1) * width;
    let k_range = key_width + key_head * width..key_width + (key_head + 1) * width;
    for (token, token_mixed) in mixed.iter().enumerate() {
        let row_base = token * 32;
        let query_token = &token_mixed[q_range.clone()];
        let key_token = &token_mixed[k_range.clone()];
        let four_blocks = token / 4;
        for b in 0..four_blocks {
            let p = b * 4;
            let keys = [
                &mixed[p][k_range.clone()],
                &mixed[p + 1][k_range.clone()],
                &mixed[p + 2][k_range.clone()],
                &mixed[p + 3][k_range.clone()],
            ];
            let pairs = arithmetic.gdn_dot_pair_four_rows(keys, query_token, key_token)?;
            for (i, &(q_dot, k_dot)) in pairs.iter().enumerate() {
                dots[row_base + p + i] = q_dot;
                dots[1024 + row_base + p + i] = k_dot;
            }
        }
        for (prior, prior_mixed) in mixed[..token].iter().enumerate().skip(four_blocks * 4) {
            let key_prior = &prior_mixed[k_range.clone()];
            let (q_dot, k_dot) = arithmetic.gdn_dot_pair(key_prior, query_token, key_token)?;
            dots[row_base + prior] = q_dot;
            dots[1024 + row_base + prior] = k_dot;
        }
        dots[row_base + token] = arithmetic.gdn_dot(key_token, query_token)?;
    }
    Ok(())
}

fn fill_key_head_dots_head_major(
    arithmetic: Arithmetic,
    q_head: &[f32],
    k_head: &[f32],
    count: usize,
    width: usize,
    dots: &mut [f32],
) -> Result<()> {
    for token in 0..count {
        let row_base = token * 32;
        let query_token = &q_head[token * width..(token + 1) * width];
        let key_token = &k_head[token * width..(token + 1) * width];
        let four_blocks = token / 4;
        for b in 0..four_blocks {
            let p = b * 4;
            let keys = [
                &k_head[p * width..(p + 1) * width],
                &k_head[(p + 1) * width..(p + 2) * width],
                &k_head[(p + 2) * width..(p + 3) * width],
                &k_head[(p + 3) * width..(p + 4) * width],
            ];
            let pairs = arithmetic.gdn_dot_pair_four_rows(keys, query_token, key_token)?;
            for (i, &(q_dot, k_dot)) in pairs.iter().enumerate() {
                dots[row_base + p + i] = q_dot;
                dots[1024 + row_base + p + i] = k_dot;
            }
        }
        for prior in (four_blocks * 4)..token {
            let key_prior = &k_head[prior * width..(prior + 1) * width];
            let (q_dot, k_dot) = arithmetic.gdn_dot_pair(key_prior, query_token, key_token)?;
            dots[row_base + prior] = q_dot;
            dots[1024 + row_base + prior] = k_dot;
        }
        dots[row_base + token] = arithmetic.gdn_dot(key_token, query_token)?;
    }
    Ok(())
}

#[allow(
    clippy::suboptimal_flops,
    reason = "Match the exact single-token GDN recurrence rounding order."
)]
fn exact_head_into(
    arithmetic: Arithmetic,
    previous: &[f32],
    inputs: &[Input<'_>],
    matrix: &mut [f32],
    output: &mut [f32],
) -> Result<()> {
    let Some(first) = inputs.first() else {
        return Err(invalid("empty GDN chunk"));
    };
    let width = first.key.len();
    let scale = 1.0 / small_float(width)?.sqrt();
    let row_count = previous.len() / width;
    let four_row_blocks = row_count / 4;
    for block_idx in 0..four_row_blocks {
        let r = block_idx * 4;
        let prev_rows = [
            &previous[r * width..(r + 1) * width],
            &previous[(r + 1) * width..(r + 2) * width],
            &previous[(r + 2) * width..(r + 3) * width],
            &previous[(r + 3) * width..(r + 4) * width],
        ];
        let state_slice = &mut matrix[r * width..(r + 4) * width];
        let (s01, s23) = state_slice.split_at_mut(2 * width);
        let (s0, s1) = s01.split_at_mut(width);
        let (s2, s3) = s23.split_at_mut(width);
        arithmetic.gdn_step_four_rows_tokens_from(
            prev_rows,
            [s0, s1, s2, s3],
            inputs,
            r,
            width,
            scale,
            output,
        )?;
    }
    for r in (four_row_blocks * 4)..row_count {
        let prev_row = &previous[r * width..(r + 1) * width];
        let state_row = &mut matrix[r * width..(r + 1) * width];
        arithmetic
            .gdn_step_row_tokens_from(prev_row, state_row, inputs, r, width, scale, output)?;
    }
    Ok(())
}

#[allow(
    clippy::suboptimal_flops,
    clippy::too_many_lines,
    reason = "The experimental FP32 chunk contract uses explicit separately rounded triangular arithmetic."
)]
fn head_into(
    arithmetic: Arithmetic,
    previous: &[f32],
    inputs: &[Input<'_>],
    key_dots: Option<&[f32]>,
    matrix: &mut [f32],
    output: &mut [f32],
) -> Result<()> {
    let Some(first) = inputs.first() else {
        return Err(invalid("empty GDN chunk"));
    };
    let width = first.key.len();
    let count = inputs.len();
    if count > 32
        || width == 0
        || width.checked_mul(width) != Some(previous.len())
        || matrix.len() != previous.len()
        || count.checked_mul(width) != Some(output.len())
        || inputs.iter().any(|input| {
            input.key.len() != width
                || input.query.len() != width
                || input.value.len() != width
                || !input.decay.is_finite()
                || !input.beta.is_finite()
                || !(0.0..=1.0).contains(&input.decay)
                || !(0.0..=1.0).contains(&input.beta)
        })
    {
        return Err(invalid("invalid GDN chunk geometry or gates"));
    }
    // products[t][j] is decay over (j,t], with an exact empty product of one.
    // This remains defined if a decay or a longer product underflows to zero.
    let mut products = [[0.0_f32; 32]; 32];
    let mut cumulative = [0.0_f32; 32];
    let mut cross_key = [[0.0_f32; 32]; 32];
    let mut cross_query = [[0.0_f32; 32]; 32];
    let mut decay = 1.0;
    for token in 0..count {
        decay *= inputs[token].decay;
        cumulative[token] = decay;
        products[token][token] = 1.0;
        for prior in (0..token).rev() {
            products[token][prior] = products[token][prior + 1] * inputs[prior + 1].decay;
        }
        if let Some(dots) = key_dots {
            let row_base = token * 32;
            for prior in 0..=token {
                cross_query[token][prior] = products[token][prior] * dots[row_base + prior];
                if prior < token {
                    cross_key[token][prior] =
                        products[token][prior] * dots[1024 + row_base + prior];
                }
            }
        } else {
            let four_blocks = token / 4;
            for b in 0..four_blocks {
                let p = b * 4;
                let keys = [
                    inputs[p].key,
                    inputs[p + 1].key,
                    inputs[p + 2].key,
                    inputs[p + 3].key,
                ];
                let pairs = arithmetic.gdn_dot_pair_four_rows(
                    keys,
                    inputs[token].query,
                    inputs[token].key,
                )?;
                for (i, &(q_dot, k_dot)) in pairs.iter().enumerate() {
                    let prior = p + i;
                    cross_query[token][prior] = products[token][prior] * q_dot;
                    cross_key[token][prior] = products[token][prior] * k_dot;
                }
            }
            for prior in (four_blocks * 4)..token {
                let (q_dot, k_dot) = arithmetic.gdn_dot_pair(
                    inputs[prior].key,
                    inputs[token].query,
                    inputs[token].key,
                )?;
                cross_query[token][prior] = products[token][prior] * q_dot;
                cross_key[token][prior] = products[token][prior] * k_dot;
            }
            cross_query[token][token] = products[token][token]
                * arithmetic.gdn_dot(inputs[token].key, inputs[token].query)?;
        }
    }
    let scale = 1.0 / small_float(width)?.sqrt();
    let last = count - 1;
    let last_cumulative = cumulative[last];
    let four_row_blocks = width / 4;
    for block_idx in 0..four_row_blocks {
        let base_row = block_idx * 4;
        let states = [
            &previous[base_row * width..(base_row + 1) * width],
            &previous[(base_row + 1) * width..(base_row + 2) * width],
            &previous[(base_row + 2) * width..(base_row + 3) * width],
            &previous[(base_row + 3) * width..(base_row + 4) * width],
        ];
        let mut deltas = [[0.0_f32; 32]; 4];
        for token in 0..count {
            let pairs = arithmetic.gdn_dot_pair_four_rows(
                states,
                inputs[token].key,
                inputs[token].query,
            )?;
            let cum = cumulative[token];
            let beta = inputs[token].beta;
            let out_base = token * width + base_row;
            for sub in 0..4 {
                let (key_dot, query_dot) = pairs[sub];
                let mut predicted = cum * key_dot;
                for (prior, &delta) in deltas[sub][..token].iter().enumerate() {
                    predicted += cross_key[token][prior] * delta;
                }
                deltas[sub][token] = (inputs[token].value[base_row + sub] - predicted) * beta;
                let mut value = cum * query_dot;
                for (prior, &delta) in deltas[sub][..=token].iter().enumerate() {
                    value += cross_query[token][prior] * delta;
                }
                output[out_base + sub] = value * scale;
            }
        }
        let mut scaled_deltas = [[0.0_f32; 32]; 4];
        for sub in 0..4 {
            for (prior, &delta) in deltas[sub][..count].iter().enumerate() {
                scaled_deltas[sub][prior] = products[last][prior] * delta;
            }
        }
        let matrix_slice = &mut matrix[base_row * width..(base_row + 4) * width];
        let (m01, m23) = matrix_slice.split_at_mut(2 * width);
        let (m0, m1) = m01.split_at_mut(width);
        let (m2, m3) = m23.split_at_mut(width);
        gdn_wy_update_four_rows(
            states,
            last_cumulative,
            inputs,
            &scaled_deltas,
            [m0, m1, m2, m3],
        );
    }
    for row in (four_row_blocks * 4)..width {
        let state = &previous[row * width..(row + 1) * width];
        let mut deltas = [0.0_f32; 32];
        for token in 0..count {
            let (key_dot, query_dot) =
                arithmetic.gdn_dot_pair(state, inputs[token].key, inputs[token].query)?;
            let mut predicted = cumulative[token] * key_dot;
            for (prior, &delta) in deltas[..token].iter().enumerate() {
                predicted += cross_key[token][prior] * delta;
            }
            deltas[token] = (inputs[token].value[row] - predicted) * inputs[token].beta;
            let mut value = cumulative[token] * query_dot;
            for (prior, &delta) in deltas[..=token].iter().enumerate() {
                value += cross_query[token][prior] * delta;
            }
            output[token * width + row] = value * scale;
        }
        let matrix_row = &mut matrix[row * width..(row + 1) * width];
        scale_slice_into(state, last_cumulative, matrix_row);
        for (prior, &delta) in deltas[..count].iter().enumerate() {
            let scaled_delta = products[last][prior] * delta;
            add_scaled_slice_in_place(matrix_row, inputs[prior].key, scaled_delta);
        }
    }
    finite(matrix, "chunkwise GDN state")?;
    finite(output, "chunkwise GDN output")
}

#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::suboptimal_flops,
    clippy::imprecise_flops,
    reason = "Keep unchanged graph preprocessing and the separately identified chunk algebra auditable."
)]
pub(super) fn recurrent_tile<'a>(
    layer: &'a RecurrentAttention,
    state: &mut LayerState,
    delta: &mut LayerDelta,
    geometry: &Geometry,
    arithmetic: Arithmetic,
    transforms: &Transforms,
    adapter: Option<&RankOneAdapter>,
    adapter_scale: f32,
    adapter_scratch: &mut AdapterScratch,
    projection_scratch: &mut crate::kernels::MatvecScratch,
    previous_norm: TileReuse<'a>,
    count: usize,
    check: &dyn Fn() -> Result<()>,
    tile: &mut TiledPrefill,
    mut executor: Option<&mut dyn crate::kernels::Pq2ProjectionExecutor>,
) -> Result<()> {
    let (
        LayerState::Recurrent {
            convolution: history,
            matrix: previous,
        },
        LayerDelta::Recurrent {
            convolution,
            matrix,
        },
    ) = (state, delta)
    else {
        return Err(invalid("chunkwise GDN requires recurrent state"));
    };
    check()?;
    let mut previous_weight = project_tile_pair(
        transforms,
        geometry,
        arithmetic,
        adapter,
        adapter_scale,
        adapter_scratch,
        projection_scratch,
        previous_norm,
        [&layer.qkv, &layer.gate],
        &tile.normalized[..count],
        &mut tile.mixed[..count],
        &mut tile.recurrent_gate[..count],
        &mut tile.transformed[..count],
        &mut tile.projector,
        reborrow_executor(&mut executor),
        check,
    )?;
    for (weight, outputs) in [
        (&layer.beta, &mut tile.beta),
        (&layer.alpha, &mut tile.alpha),
    ] {
        check()?;
        previous_weight = project_tile(
            transforms,
            geometry,
            arithmetic,
            adapter,
            adapter_scale,
            adapter_scratch,
            projection_scratch,
            previous_weight,
            weight,
            &tile.normalized[..count],
            &mut outputs[..count],
            &mut tile.transformed[..count],
            &mut tile.projector,
            reborrow_executor(&mut executor),
            check,
        )?;
    }
    copy_into(history, convolution)?;
    resize_into(previous.len(), matrix)?;
    let width = geometry.ssm_head;
    let hist_width = geometry.convolution - 1;
    let conv_aligned = hist_width > 0
        && geometry.convolution_width().is_multiple_of(width)
        && width.is_multiple_of(4);
    let parallelize_conv = conv_aligned && tile.projector.row_parallelism().workers() > 1;
    let parallelize_post_conv = tile.projector.row_parallelism().workers() > 1 && count > 1;
    let conv_head_major = conv_aligned && count > 1;
    if conv_head_major {
        for attended in tile.attended.iter_mut().take(count) {
            check()?;
            resize_into(geometry.value_width(), attended)?;
            check()?;
        }
        let head_hist = width * hist_width;
        let head_coeff = width * geometry.convolution;
        let head_out_len = count * width;
        let num_conv_heads = geometry.convolution_width() / width;
        let qk_heads = (2 * geometry.key_width()) / width;
        resize_into(num_conv_heads * head_out_len, &mut tile.gdn_conv_outputs)?;
        let mixed_ref: &[Vec<f32>] = &tile.mixed[..count];
        tile.projector.row_parallelism().update_paired_chunks(
            convolution,
            head_hist,
            &mut tile.gdn_conv_outputs,
            head_out_len,
            |base_head, conv_chunk, out_chunk| {
                for (offset, (head_conv, head_out)) in conv_chunk
                    .chunks_exact_mut(head_hist)
                    .zip(out_chunk.chunks_exact_mut(head_out_len))
                    .enumerate()
                {
                    let head = base_head + offset;
                    let c_slice = &layer.convolution[head * head_coeff..(head + 1) * head_coeff];
                    for (token, token_mixed) in mixed_ref.iter().enumerate() {
                        head_out[token * width..(token + 1) * width]
                            .copy_from_slice(&token_mixed[head * width..(head + 1) * width]);
                    }
                    if matches!(arithmetic, Arithmetic::PinnedSwift) {
                        crate::kernels::pinned_convolution::step_silu_tokens_in_place(
                            head_conv,
                            head_out,
                            width,
                            c_slice,
                            geometry.convolution,
                        )?;
                    } else {
                        for head_mixed in head_out.chunks_exact_mut(width) {
                            for (channel, value) in head_mixed.iter_mut().enumerate() {
                                let projected = *value;
                                let start = channel * hist_width;
                                let coefficients = &c_slice[channel * geometry.convolution
                                    ..(channel + 1) * geometry.convolution];
                                *value = arithmetic.convolution_row(
                                    &head_conv[start..start + hist_width],
                                    projected,
                                    coefficients,
                                )?;
                                head_conv[start..start + hist_width].rotate_left(1);
                                head_conv[start + hist_width - 1] = projected;
                            }
                            arithmetic.silu(head_mixed);
                        }
                    }
                    if head < qk_heads {
                        for head_mixed in head_out.chunks_exact_mut(width) {
                            l2_norm(head_mixed, geometry.epsilon);
                        }
                    }
                }
                Ok(())
            },
        )?;
    } else {
        for token in 0..count {
            check()?;
            let mixed = &mut tile.mixed[token];
            if parallelize_conv {
                let head_hist = width * hist_width;
                let head_coeff = width * geometry.convolution;
                let qk_heads = (2 * geometry.key_width()) / width;
                tile.projector.row_parallelism().update_paired_chunks(
                    mixed,
                    width,
                    convolution,
                    head_hist,
                    |base_head, mixed_chunk, conv_chunk| {
                        for (offset, (head_mixed, head_conv)) in mixed_chunk
                            .chunks_exact_mut(width)
                            .zip(conv_chunk.chunks_exact_mut(head_hist))
                            .enumerate()
                        {
                            let head = base_head + offset;
                            let c_slice =
                                &layer.convolution[head * head_coeff..(head + 1) * head_coeff];
                            if matches!(arithmetic, Arithmetic::PinnedSwift) {
                                crate::kernels::pinned_convolution::step_silu_in_place(
                                    head_conv,
                                    head_mixed,
                                    c_slice,
                                    geometry.convolution,
                                )?;
                            } else {
                                for (channel, value) in head_mixed.iter_mut().enumerate() {
                                    let projected = *value;
                                    let start = channel * hist_width;
                                    let coefficients = &c_slice[channel * geometry.convolution
                                        ..(channel + 1) * geometry.convolution];
                                    *value = arithmetic.convolution_row(
                                        &head_conv[start..start + hist_width],
                                        projected,
                                        coefficients,
                                    )?;
                                    head_conv[start..start + hist_width].rotate_left(1);
                                    head_conv[start + hist_width - 1] = projected;
                                }
                                arithmetic.silu(head_mixed);
                            }
                            if head < qk_heads {
                                l2_norm(head_mixed, geometry.epsilon);
                            }
                        }
                        Ok(())
                    },
                )?;
            } else if matches!(arithmetic, Arithmetic::PinnedSwift) {
                crate::kernels::pinned_convolution::step_silu_in_place(
                    convolution,
                    mixed,
                    &layer.convolution,
                    geometry.convolution,
                )?;
            } else {
                for (channel, value) in mixed.iter_mut().enumerate() {
                    let projected = *value;
                    let start = channel * hist_width;
                    let coefficients = &layer.convolution
                        [channel * geometry.convolution..(channel + 1) * geometry.convolution];
                    *value = arithmetic.convolution_row(
                        &convolution[start..start + hist_width],
                        projected,
                        coefficients,
                    )?;
                    if geometry.convolution > 1 {
                        convolution[start..start + hist_width].rotate_left(1);
                        convolution[start + hist_width - 1] = projected;
                    }
                }
                arithmetic.silu(mixed);
            }
            resize_into(geometry.value_width(), &mut tile.attended[token])?;
            if !parallelize_post_conv {
                if !parallelize_conv {
                    for head in mixed[..2 * geometry.key_width()].chunks_exact_mut(width) {
                        l2_norm(head, geometry.epsilon);
                    }
                }
                for head in 0..geometry.ssm_values {
                    let activation = tile.alpha[token][head] + layer.time_bias[head];
                    let softplus = if activation > 20.0 {
                        activation
                    } else {
                        (1.0 + activation.exp()).ln()
                    };
                    tile.alpha[token][head] = (softplus * layer.decay[head]).exp();
                    tile.beta[token][head] = sigmoid(tile.beta[token][head]);
                }
            }
            check()?;
        }
    }
    if conv_head_major || parallelize_post_conv {
        if !conv_head_major && !parallelize_conv {
            tile.projector.row_parallelism().update_chunks(
                &mut tile.mixed[..count],
                1,
                |_, mixed_chunk| {
                    for mixed in mixed_chunk {
                        for head in mixed[..2 * geometry.key_width()].chunks_exact_mut(width) {
                            l2_norm(head, geometry.epsilon);
                        }
                    }
                    Ok(())
                },
            )?;
        }
        tile.projector.row_parallelism().update_paired_chunks(
            &mut tile.alpha[..count],
            1,
            &mut tile.beta[..count],
            1,
            |_, alpha_chunk, beta_chunk| {
                for (alpha, beta) in alpha_chunk.iter_mut().zip(beta_chunk.iter_mut()) {
                    for head in 0..geometry.ssm_values {
                        let activation = alpha[head] + layer.time_bias[head];
                        let softplus = if activation > 20.0 {
                            activation
                        } else {
                            (1.0 + activation.exp()).ln()
                        };
                        alpha[head] = (softplus * layer.decay[head]).exp();
                        beta[head] = sigmoid(beta[head]);
                    }
                }
                Ok(())
            },
        )?;
    }
    let chunkwise = tile.chunkwise_gdn;
    let use_shared_key_dots = chunkwise
        && count <= 32
        && width > 0
        && geometry.ssm_values > geometry.ssm_keys
        && geometry.ssm_keys > 0;
    if use_shared_key_dots {
        resize_into(geometry.ssm_keys * KEY_DOT_STRIDE, &mut tile.gdn_key_dots)?;
        let mixed_ref: &[Vec<f32>] = &tile.mixed[..count];
        let conv_ref: &[f32] = &tile.gdn_conv_outputs;
        let key_width = geometry.key_width();
        let head_out_len = count * width;
        tile.projector.row_parallelism().update_chunks(
            &mut tile.gdn_key_dots,
            KEY_DOT_STRIDE,
            |base_key, chunk| {
                for (offset, dots) in chunk
                    .as_chunks_mut::<KEY_DOT_STRIDE>()
                    .0
                    .iter_mut()
                    .enumerate()
                {
                    let key_head = base_key + offset;
                    if conv_head_major {
                        let q_head =
                            &conv_ref[key_head * head_out_len..(key_head + 1) * head_out_len];
                        let k_head_idx = geometry.ssm_keys + key_head;
                        let k_head =
                            &conv_ref[k_head_idx * head_out_len..(k_head_idx + 1) * head_out_len];
                        fill_key_head_dots_head_major(
                            arithmetic, q_head, k_head, count, width, dots,
                        )?;
                    } else {
                        fill_key_head_dots(
                            arithmetic, mixed_ref, key_width, width, key_head, dots,
                        )?;
                    }
                }
                Ok(())
            },
        )?;
    }
    if tile.projector.row_parallelism().workers() > 1 && geometry.ssm_values > 1 {
        if chunkwise {
            for _ in 0..geometry.ssm_values {
                check()?;
            }
        }
        let head_state_len = width * width;
        let head_out_len = count * width;
        resize_into(
            geometry.ssm_values * head_out_len,
            &mut tile.gdn_head_outputs,
        )?;
        let mixed_ref: &[Vec<f32>] = &tile.mixed[..count];
        let conv_ref: &[f32] = &tile.gdn_conv_outputs;
        let alpha_ref: &[Vec<f32>] = &tile.alpha[..count];
        let beta_ref: &[Vec<f32>] = &tile.beta[..count];
        let gate_ref: &[Vec<f32>] = &tile.recurrent_gate[..count];
        let prev_ref: &[f32] = previous;
        let key_dots_ref: &[f32] = &tile.gdn_key_dots;
        tile.projector.row_parallelism().update_paired_chunks(
            matrix,
            head_state_len,
            &mut tile.gdn_head_outputs,
            head_out_len,
            |base_head, matrix_chunk, out_chunk| {
                let mut inputs: [Input<'_>; 32] = std::array::from_fn(|_| Input {
                    query: &[],
                    key: &[],
                    value: &[],
                    decay: 0.0,
                    beta: 0.0,
                });
                for (offset, (head_matrix, head_out)) in matrix_chunk
                    .chunks_exact_mut(head_state_len)
                    .zip(out_chunk.chunks_exact_mut(head_out_len))
                    .enumerate()
                {
                    let head_index = base_head + offset;
                    let key_head = head_index % geometry.ssm_keys;
                    if conv_head_major {
                        let q_head =
                            &conv_ref[key_head * head_out_len..(key_head + 1) * head_out_len];
                        let k_head_idx = geometry.ssm_keys + key_head;
                        let k_head =
                            &conv_ref[k_head_idx * head_out_len..(k_head_idx + 1) * head_out_len];
                        let v_head_idx = 2 * geometry.ssm_keys + head_index;
                        let v_head =
                            &conv_ref[v_head_idx * head_out_len..(v_head_idx + 1) * head_out_len];
                        for (token, slot) in inputs[..count].iter_mut().enumerate() {
                            *slot = Input {
                                query: &q_head[token * width..(token + 1) * width],
                                key: &k_head[token * width..(token + 1) * width],
                                value: &v_head[token * width..(token + 1) * width],
                                decay: alpha_ref[token][head_index],
                                beta: beta_ref[token][head_index],
                            };
                        }
                    } else {
                        for (token, slot) in inputs[..count].iter_mut().enumerate() {
                            let mixed = &mixed_ref[token];
                            *slot = Input {
                                query: &mixed[key_head * width..(key_head + 1) * width],
                                key: &mixed[geometry.key_width() + key_head * width
                                    ..geometry.key_width() + (key_head + 1) * width],
                                value: &mixed[2 * geometry.key_width() + head_index * width
                                    ..2 * geometry.key_width() + (head_index + 1) * width],
                                decay: alpha_ref[token][head_index],
                                beta: beta_ref[token][head_index],
                            };
                        }
                    }
                    let prev_head =
                        &prev_ref[head_index * head_state_len..(head_index + 1) * head_state_len];
                    if chunkwise {
                        let head_dots = use_shared_key_dots.then(|| {
                            &key_dots_ref
                                [key_head * KEY_DOT_STRIDE..(key_head + 1) * KEY_DOT_STRIDE]
                        });
                        head_into(
                            arithmetic,
                            prev_head,
                            &inputs[..count],
                            head_dots,
                            head_matrix,
                            head_out,
                        )?;
                    } else {
                        exact_head_into(
                            arithmetic,
                            prev_head,
                            &inputs[..count],
                            head_matrix,
                            head_out,
                        )?;
                    }
                    for token in 0..count {
                        let token_out = &mut head_out[token * width..(token + 1) * width];
                        arithmetic.normalize_in_place(token_out, &layer.norm, geometry.epsilon)?;
                        arithmetic.swiglu_in_place(
                            &gate_ref[token][head_index * width..(head_index + 1) * width],
                            token_out,
                        )?;
                    }
                }
                Ok(())
            },
        )?;
        for head_index in 0..geometry.ssm_values {
            let head_out =
                &tile.gdn_head_outputs[head_index * head_out_len..(head_index + 1) * head_out_len];
            for token in 0..count {
                tile.attended[token][head_index * width..(head_index + 1) * width]
                    .copy_from_slice(&head_out[token * width..(token + 1) * width]);
            }
        }
        if chunkwise {
            tile.statistics.chunkwise_gdn_heads = tile
                .statistics
                .chunkwise_gdn_heads
                .saturating_add(u64::try_from(geometry.ssm_values)?);
        }
    } else {
        let mut inputs: [Input<'_>; 32] = std::array::from_fn(|_| Input {
            query: &[],
            key: &[],
            value: &[],
            decay: 0.0,
            beta: 0.0,
        });
        let head_out_len = count * width;
        resize_into(head_out_len, &mut tile.gdn_head_outputs)?;
        for head_index in 0..geometry.ssm_values {
            if chunkwise {
                check()?;
            }
            let key_head = head_index % geometry.ssm_keys;
            if conv_head_major {
                let conv_ref: &[f32] = &tile.gdn_conv_outputs;
                let q_head = &conv_ref[key_head * head_out_len..(key_head + 1) * head_out_len];
                let k_head_idx = geometry.ssm_keys + key_head;
                let k_head = &conv_ref[k_head_idx * head_out_len..(k_head_idx + 1) * head_out_len];
                let v_head_idx = 2 * geometry.ssm_keys + head_index;
                let v_head = &conv_ref[v_head_idx * head_out_len..(v_head_idx + 1) * head_out_len];
                for (token, slot) in inputs[..count].iter_mut().enumerate() {
                    *slot = Input {
                        query: &q_head[token * width..(token + 1) * width],
                        key: &k_head[token * width..(token + 1) * width],
                        value: &v_head[token * width..(token + 1) * width],
                        decay: tile.alpha[token][head_index],
                        beta: tile.beta[token][head_index],
                    };
                }
            } else {
                for (token, slot) in inputs[..count].iter_mut().enumerate() {
                    let mixed = &tile.mixed[token];
                    *slot = Input {
                        query: &mixed[key_head * width..(key_head + 1) * width],
                        key: &mixed[geometry.key_width() + key_head * width
                            ..geometry.key_width() + (key_head + 1) * width],
                        value: &mixed[2 * geometry.key_width() + head_index * width
                            ..2 * geometry.key_width() + (head_index + 1) * width],
                        decay: tile.alpha[token][head_index],
                        beta: tile.beta[token][head_index],
                    };
                }
            }
            let range = head_index * width * width..(head_index + 1) * width * width;
            if chunkwise {
                let head_dots = use_shared_key_dots.then(|| {
                    &tile.gdn_key_dots[key_head * KEY_DOT_STRIDE..(key_head + 1) * KEY_DOT_STRIDE]
                });
                head_into(
                    arithmetic,
                    &previous[range.clone()],
                    &inputs[..count],
                    head_dots,
                    &mut matrix[range],
                    &mut tile.gdn_head_outputs,
                )?;
            } else {
                exact_head_into(
                    arithmetic,
                    &previous[range.clone()],
                    &inputs[..count],
                    &mut matrix[range],
                    &mut tile.gdn_head_outputs,
                )?;
            }
            for token in 0..count {
                let token_out = &mut tile.gdn_head_outputs[token * width..(token + 1) * width];
                arithmetic.normalize_in_place(token_out, &layer.norm, geometry.epsilon)?;
                arithmetic.swiglu_into(
                    &tile.recurrent_gate[token][head_index * width..(head_index + 1) * width],
                    token_out,
                    &mut tile.attended[token][head_index * width..(head_index + 1) * width],
                )?;
            }
            if chunkwise {
                tile.statistics.chunkwise_gdn_heads =
                    tile.statistics.chunkwise_gdn_heads.saturating_add(1);
            }
        }
    }
    check()?;
    let _ = project_tile(
        transforms,
        geometry,
        arithmetic,
        adapter,
        adapter_scale,
        adapter_scratch,
        projection_scratch,
        TileReuse::default(),
        &layer.output,
        &tile.attended[..count],
        &mut tile.attention[..count],
        &mut tile.transformed[..count],
        &mut tile.projector,
        reborrow_executor(&mut executor),
        check,
    )?;
    finite(convolution, "convolution state")?;
    std::mem::swap(history, convolution);
    std::mem::swap(previous, matrix);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(
        clippy::suboptimal_flops,
        reason = "Independent ordinary recurrence uses explicit multiply/add rather than chunk identities."
    )]
    fn chunkwise_gdn_triangular_solve_matches_sequential_signed_and_zero_decay() -> Result<()> {
        let queries = [[0.5, -0.25], [-0.25, 0.5], [0.25, 0.125], [-0.5, -0.25]];
        let keys = [[0.25, -0.5], [-0.5, 0.25], [0.5, 0.25], [-0.125, -0.5]];
        let values = [[0.5, -0.75], [-1.0, 0.25], [0.125, -0.5], [-0.75, -0.125]];
        for decays in [[0.5, 0.75, 0.25, 0.875], [0.5, 0.0, 0.25, 0.875], [0.0; 4]] {
            let inputs: Vec<_> = (0..4)
                .map(|token| Input {
                    query: &queries[token],
                    key: &keys[token],
                    value: &values[token],
                    decay: decays[token],
                    beta: [0.125, 0.25, 0.75, 0.5][token],
                })
                .collect();
            let previous = [0.125, -0.25, -0.5, 0.75];
            let mut expected = previous;
            let mut expected_outputs = Vec::new();
            for input in &inputs {
                for element in &mut expected {
                    *element *= input.decay;
                }
                for (row, &value) in expected.as_chunks_mut::<2>().0.iter_mut().zip(input.value) {
                    let predicted = row[0] * input.key[0] + row[1] * input.key[1];
                    let delta = (value - predicted) * input.beta;
                    for (element, &key) in row.iter_mut().zip(input.key) {
                        *element += delta * key;
                    }
                    expected_outputs
                        .push((row[0] * input.query[0] + row[1] * input.query[1]) / 2.0_f32.sqrt());
                }
            }
            for arithmetic in [Arithmetic::Source, Arithmetic::PinnedSwift] {
                let mut actual = [0.0; 4];
                let mut outputs = Vec::new();
                head(arithmetic, &previous, &inputs, &mut actual, &mut outputs)?;
                for (&actual, &expected) in actual
                    .iter()
                    .chain(&outputs)
                    .zip(expected.iter().chain(&expected_outputs))
                {
                    assert!((actual - expected).abs() <= 0.000_001);
                }
            }
        }
        Ok(())
    }

    #[test]
    fn chunkwise_gdn_rejects_invalid_shapes_and_gates() {
        let input = Input {
            query: &[1.0],
            key: &[1.0],
            value: &[1.0],
            decay: -0.5,
            beta: 0.5,
        };
        assert!(
            head(
                Arithmetic::Source,
                &[0.0],
                &[input],
                &mut [0.0],
                &mut Vec::new()
            )
            .is_err()
        );
        assert!(head(Arithmetic::Source, &[], &[], &mut [], &mut Vec::new()).is_err());
    }

    #[test]
    #[allow(
        clippy::suboptimal_flops,
        reason = "Verifies bit-for-bit parity against the single-row WY reference."
    )]
    fn chunkwise_gdn_four_row_and_vector_update_matches_single_row_bits() -> Result<()> {
        for width in [6, 16, 18, 32] {
            let count = 6;
            let q_buf: Vec<f32> = (0..count * width)
                .map(|i| {
                    let s = i16::try_from((i * 37 + 11) % 64).unwrap_or(0) - 32;
                    f32::from(s) * 0.03125
                })
                .collect();
            let k_buf: Vec<f32> = (0..count * width)
                .map(|i| {
                    let s = i16::try_from((i * 53 + 7) % 64).unwrap_or(0) - 32;
                    f32::from(s) * 0.03125
                })
                .collect();
            let v_buf: Vec<f32> = (0..count * width)
                .map(|i| {
                    let s = i16::try_from((i * 29 + 3) % 64).unwrap_or(0) - 32;
                    f32::from(s) * 0.03125
                })
                .collect();
            let prev: Vec<f32> = (0..width * width)
                .map(|i| {
                    let s = i16::try_from((i * 19 + 5) % 64).unwrap_or(0) - 32;
                    f32::from(s) * 0.015_625
                })
                .collect();
            let decays = [0.875_f32, 0.75, 0.625, 0.5, 0.9375, 0.8125];
            let betas = [0.25_f32, 0.5, 0.375, 0.625, 0.75, 0.125];
            let inputs: Vec<_> = (0..count)
                .map(|t| Input {
                    query: &q_buf[t * width..(t + 1) * width],
                    key: &k_buf[t * width..(t + 1) * width],
                    value: &v_buf[t * width..(t + 1) * width],
                    decay: decays[t],
                    beta: betas[t],
                })
                .collect();

            for arithmetic in [Arithmetic::Source, Arithmetic::PinnedSwift] {
                let mut actual_mat = vec![0.0_f32; width * width];
                let mut actual_out = Vec::new();
                head(arithmetic, &prev, &inputs, &mut actual_mat, &mut actual_out)?;

                let mut ref_mat = vec![0.0_f32; width * width];
                for (r, state) in prev.chunks_exact(width).enumerate() {
                    let mut deltas = [0.0_f32; 32];
                    let mut cum = 1.0_f32;
                    for t in 0..count {
                        cum *= inputs[t].decay;
                        let (key_dot, _) =
                            arithmetic.gdn_dot_pair(state, inputs[t].key, inputs[t].query)?;
                        let mut predicted = cum * key_dot;
                        for (p, &d) in deltas[..t].iter().enumerate() {
                            let mut prod = 1.0_f32;
                            for step in ((p + 1)..=t).rev() {
                                prod *= inputs[step].decay;
                            }
                            let (_, k_dot) = arithmetic.gdn_dot_pair(
                                inputs[p].key,
                                inputs[t].query,
                                inputs[t].key,
                            )?;
                            predicted += (prod * k_dot) * d;
                        }
                        deltas[t] = (inputs[t].value[r] - predicted) * inputs[t].beta;
                    }
                    let row_slice = &mut ref_mat[r * width..(r + 1) * width];
                    scale_slice_into(state, cum, row_slice);
                    for (p, &d) in deltas[..count].iter().enumerate() {
                        let mut prod = 1.0_f32;
                        for step in ((p + 1)..count).rev() {
                            prod *= inputs[step].decay;
                        }
                        add_scaled_slice_in_place(row_slice, inputs[p].key, prod * d);
                    }
                }
                for (a, b) in actual_mat.iter().zip(&ref_mat) {
                    assert_eq!(a.to_bits(), b.to_bits());
                }
            }
        }
        Ok(())
    }
}
