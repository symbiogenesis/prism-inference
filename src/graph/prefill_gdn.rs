//! Opt-in bounded triangular GDN chunks; no cumulative-decay division.

use super::{TiledPrefill, project_tile, project_tile_pair, reborrow_executor};
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
    head_into(arithmetic, previous, inputs, matrix, output)
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
        for prior in 0..=token {
            cross_query[token][prior] = products[token][prior]
                * arithmetic.gdn_dot(inputs[prior].key, inputs[token].query)?;
            if prior < token {
                cross_key[token][prior] = products[token][prior]
                    * arithmetic.gdn_dot(inputs[prior].key, inputs[token].key)?;
            }
        }
    }
    let scale = 1.0 / small_float(width)?.sqrt();
    for (row, state) in previous.chunks_exact(width).enumerate() {
        let mut deltas = [0.0_f32; 32];
        for token in 0..count {
            let mut predicted = cumulative[token] * arithmetic.gdn_dot(state, inputs[token].key)?;
            for (prior, &delta) in deltas[..token].iter().enumerate() {
                predicted += cross_key[token][prior] * delta;
            }
            deltas[token] = (inputs[token].value[row] - predicted) * inputs[token].beta;
            let mut value = cumulative[token] * arithmetic.gdn_dot(state, inputs[token].query)?;
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
        project_tile(
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
        previous_weight = Some(weight);
    }
    copy_into(history, convolution)?;
    resize_into(previous.len(), matrix)?;
    for token in 0..count {
        check()?;
        let mixed = &mut tile.mixed[token];
        if matches!(arithmetic, Arithmetic::PinnedSwift) {
            crate::kernels::pinned_convolution::step_silu_in_place(
                convolution,
                mixed,
                &layer.convolution,
                geometry.convolution,
            )?;
        } else {
            for (channel, value) in mixed.iter_mut().enumerate() {
                let projected = *value;
                let start = channel * (geometry.convolution - 1);
                let coefficients = &layer.convolution
                    [channel * geometry.convolution..(channel + 1) * geometry.convolution];
                *value = arithmetic.convolution_row(
                    &convolution[start..start + geometry.convolution - 1],
                    projected,
                    coefficients,
                )?;
                if geometry.convolution > 1 {
                    convolution[start..start + geometry.convolution - 1].rotate_left(1);
                    convolution[start + geometry.convolution - 2] = projected;
                }
            }
            arithmetic.silu(mixed);
        }
        for head in mixed[..2 * geometry.key_width()].chunks_exact_mut(geometry.ssm_head) {
            l2_norm(head, geometry.epsilon);
        }
        resize_into(geometry.value_width(), &mut tile.attended[token])?;
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
        check()?;
    }
    let width = geometry.ssm_head;
    if tile.projector.row_parallelism().workers() > 1 && geometry.ssm_values > 1 {
        for _ in 0..geometry.ssm_values {
            check()?;
        }
        let head_state_len = width * width;
        let head_out_len = count * width;
        let mut head_outputs = Vec::new();
        resize_into(geometry.ssm_values * head_out_len, &mut head_outputs)?;
        let mixed_ref: &[Vec<f32>] = &tile.mixed[..count];
        let alpha_ref: &[Vec<f32>] = &tile.alpha[..count];
        let beta_ref: &[Vec<f32>] = &tile.beta[..count];
        let gate_ref: &[Vec<f32>] = &tile.recurrent_gate[..count];
        let prev_ref: &[f32] = previous;
        tile.projector.row_parallelism().update_paired_chunks(
            matrix,
            head_state_len,
            &mut head_outputs,
            head_out_len,
            |base_head, matrix_chunk, out_chunk| {
                let mut inputs = Vec::new();
                inputs.try_reserve_exact(count)?;
                for (offset, (head_matrix, head_out)) in matrix_chunk
                    .chunks_exact_mut(head_state_len)
                    .zip(out_chunk.chunks_exact_mut(head_out_len))
                    .enumerate()
                {
                    let head_index = base_head + offset;
                    let key_head = head_index % geometry.ssm_keys;
                    inputs.clear();
                    for token in 0..count {
                        let mixed = &mixed_ref[token];
                        inputs.push(Input {
                            query: &mixed[key_head * width..(key_head + 1) * width],
                            key: &mixed[geometry.key_width() + key_head * width
                                ..geometry.key_width() + (key_head + 1) * width],
                            value: &mixed[2 * geometry.key_width() + head_index * width
                                ..2 * geometry.key_width() + (head_index + 1) * width],
                            decay: alpha_ref[token][head_index],
                            beta: beta_ref[token][head_index],
                        });
                    }
                    let prev_head =
                        &prev_ref[head_index * head_state_len..(head_index + 1) * head_state_len];
                    head_into(arithmetic, prev_head, &inputs, head_matrix, head_out)?;
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
                &head_outputs[head_index * head_out_len..(head_index + 1) * head_out_len];
            for token in 0..count {
                tile.attended[token][head_index * width..(head_index + 1) * width]
                    .copy_from_slice(&head_out[token * width..(token + 1) * width]);
            }
        }
        tile.statistics.chunkwise_gdn_heads = tile
            .statistics
            .chunkwise_gdn_heads
            .saturating_add(u64::try_from(geometry.ssm_values)?);
    } else {
        let mut inputs = Vec::new();
        inputs.try_reserve_exact(count)?;
        let mut output = Vec::new();
        let mut normalized = Vec::new();
        for head_index in 0..geometry.ssm_values {
            check()?;
            inputs.clear();
            let key_head = head_index % geometry.ssm_keys;
            for token in 0..count {
                let mixed = &tile.mixed[token];
                inputs.push(Input {
                    query: &mixed[key_head * width..(key_head + 1) * width],
                    key: &mixed[geometry.key_width() + key_head * width
                        ..geometry.key_width() + (key_head + 1) * width],
                    value: &mixed[2 * geometry.key_width() + head_index * width
                        ..2 * geometry.key_width() + (head_index + 1) * width],
                    decay: tile.alpha[token][head_index],
                    beta: tile.beta[token][head_index],
                });
            }
            let range = head_index * width * width..(head_index + 1) * width * width;
            head(
                arithmetic,
                &previous[range.clone()],
                &inputs,
                &mut matrix[range],
                &mut output,
            )?;
            for token in 0..count {
                arithmetic.normalize(
                    &output[token * width..(token + 1) * width],
                    &layer.norm,
                    geometry.epsilon,
                    &mut normalized,
                )?;
                arithmetic.swiglu_into(
                    &tile.recurrent_gate[token][head_index * width..(head_index + 1) * width],
                    &normalized,
                    &mut tile.attended[token][head_index * width..(head_index + 1) * width],
                )?;
            }
            tile.statistics.chunkwise_gdn_heads =
                tile.statistics.chunkwise_gdn_heads.saturating_add(1);
        }
    }
    check()?;
    project_tile(
        transforms,
        geometry,
        arithmetic,
        adapter,
        adapter_scale,
        adapter_scratch,
        projection_scratch,
        None,
        &layer.output,
        &tile.attended[..count],
        &mut tile.attention[..count],
        &mut tile.transformed[..count],
        &mut tile.projector,
        reborrow_executor(&mut executor),
        check,
    )?;
    finite(convolution, "chunkwise convolution state")?;
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
