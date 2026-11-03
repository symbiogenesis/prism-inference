//! Opt-in bounded triangular GDN chunks; no cumulative-decay division.

use super::{TiledPrefill, project_tile};
use crate::graph::{
    Arithmetic, Geometry, LayerDelta, LayerState, RecurrentAttention, Transforms, copy_into,
    finite, l2_norm, sigmoid, small_float, zeros_into,
};
use crate::model::lora::{RankOneAdapter, Scratch as AdapterScratch};
use crate::{Result, invalid};

struct Input<'a> {
    query: &'a [f32],
    key: &'a [f32],
    value: &'a [f32],
    decay: f32,
    beta: f32,
}

#[allow(
    clippy::suboptimal_flops,
    clippy::too_many_lines,
    reason = "The experimental FP32 chunk contract uses explicit separately rounded triangular arithmetic."
)]
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
    if count > 32
        || width == 0
        || width.checked_mul(width) != Some(previous.len())
        || matrix.len() != previous.len()
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
    zeros_into(
        count
            .checked_mul(width)
            .ok_or_else(|| invalid("GDN chunk output overflow"))?,
        output,
    )?;
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
        for (column, &initial) in state.iter().enumerate() {
            let mut value = cumulative[last] * initial;
            for (prior, &delta) in deltas[..count].iter().enumerate() {
                value += (products[last][prior] * delta) * inputs[prior].key[column];
            }
            matrix[row * width + column] = value;
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
    count: usize,
    check: &dyn Fn() -> Result<()>,
    tile: &mut TiledPrefill,
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
    for (weight, outputs) in [
        (&layer.qkv, &mut tile.mixed),
        (&layer.gate, &mut tile.recurrent_gate),
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
            weight,
            &tile.normalized[..count],
            &mut outputs[..count],
            &mut tile.transformed[..count],
            &mut tile.projector,
            check,
        )?;
    }
    copy_into(history, convolution)?;
    copy_into(previous, matrix)?;
    for token in 0..count {
        check()?;
        let mixed = &mut tile.mixed[token];
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
        for head in mixed[..2 * geometry.key_width()].chunks_exact_mut(geometry.ssm_head) {
            l2_norm(head, geometry.epsilon);
        }
        zeros_into(geometry.value_width(), &mut tile.attended[token])?;
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
    let mut inputs = Vec::new();
    inputs.try_reserve_exact(count)?;
    let mut output = Vec::new();
    let mut normalized = Vec::new();
    for head_index in 0..geometry.ssm_values {
        check()?;
        inputs.clear();
        let key_head = head_index % geometry.ssm_keys;
        let width = geometry.ssm_head;
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
        tile.statistics.chunkwise_gdn_heads = tile.statistics.chunkwise_gdn_heads.saturating_add(1);
    }
    check()?;
    project_tile(
        transforms,
        geometry,
        arithmetic,
        adapter,
        adapter_scale,
        adapter_scratch,
        &layer.output,
        &tile.attended[..count],
        &mut tile.attention[..count],
        &mut tile.transformed[..count],
        &mut tile.projector,
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
                for (row, &value) in expected.chunks_exact_mut(2).zip(input.value) {
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
