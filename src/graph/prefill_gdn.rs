//! Opt-in bounded triangular GDN chunks; no cumulative-decay division.

use super::{TileReuse, TiledPrefill, project_tile, project_tile_pair, reborrow_executor};
use crate::graph::{
    Arithmetic, Geometry, LayerDelta, LayerState, RecurrentAttention, Transforms, copy_into,
    finite, l2_norm, resize_into, sigmoid, small_float,
};
use crate::kernels::{add_scaled_slice_in_place, scale_slice_into};
use crate::model::lora::{RankOneAdapter, Scratch as AdapterScratch};
use crate::{Result, invalid};

struct Input<'a> {
    query: &'a [f32],
    key: &'a [f32],
    value: &'a [f32],
    decay: f32,
    beta: f32,
}

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
        for (prior, prior_mixed) in mixed[..token].iter().enumerate() {
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
        for prior in 0..token {
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
    for ((row, prev_row), state_row) in previous
        .chunks_exact(width)
        .enumerate()
        .zip(matrix.chunks_exact_mut(width))
    {
        let first_dot_k =
            arithmetic.gdn_decay_and_dot_from(prev_row, state_row, first.key, first.decay)?;
        let first_delta = (first.value[row] - first_dot_k) * first.beta;
        output[row] =
            arithmetic.gdn_update_and_dot(state_row, first.key, first_delta, first.query)? * scale;
        for (token, input) in inputs.iter().enumerate().skip(1) {
            let dot_k = arithmetic.gdn_decay_and_dot(state_row, input.key, input.decay)?;
            let delta = (input.value[row] - dot_k) * input.beta;
            output[token * width + row] =
                arithmetic.gdn_update_and_dot(state_row, input.key, delta, input.query)? * scale;
        }
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
            for prior in 0..token {
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
    for (row, state) in previous.chunks_exact(width).enumerate() {
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
        let last = count - 1;
        let last_cumulative = cumulative[last];
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
pub(super) fn recurrent_tile(
    layer: &RecurrentAttention,
    state: &mut LayerState,
    delta: &mut LayerDelta,
    geometry: &Geometry,
    arithmetic: Arithmetic,
    transforms: &Transforms,
    adapter: Option<&RankOneAdapter>,
    adapter_scale: f32,
    adapter_scratch: &mut AdapterScratch,
    projection_scratch: &mut crate::kernels::MatvecScratch,
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
    let parallelize_conv = hist_width > 0
        && tile.projector.row_parallelism().workers() > 1
        && geometry.convolution_width().is_multiple_of(width)
        && width.is_multiple_of(4);
    let parallelize_post_conv = tile.projector.row_parallelism().workers() > 1 && count > 1;
    let conv_head_major = parallelize_conv && count > 1;
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
                        let head_mixed = &mut head_out[token * width..(token + 1) * width];
                        head_mixed.copy_from_slice(&token_mixed[head * width..(head + 1) * width]);
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
    if parallelize_post_conv {
        if !parallelize_conv {
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
}
