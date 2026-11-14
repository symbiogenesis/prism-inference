//! Actual recurrent-layer integration against defensive per-row/per-head kernels.

use super::{
    Arithmetic, Evaluation, Geometry, LayerDelta, ProjectionScratch, RecurrentAttention,
    Transforms, l2_norm, sigmoid, small_float,
};
use crate::{Result, kernels, model::Tensor};
use std::sync::Arc;

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

fn placeholder() -> Tensor {
    Tensor {
        name: "unused-direct-recurrent-projection".to_owned(),
        dimensions: vec![1, 1],
        kind: 0,
        data: Arc::from(0.0_f32.to_le_bytes()),
    }
}

#[expect(
    clippy::imprecise_flops,
    reason = "Keep the defensive recurrence's pinned logf(1+expf(x)) gate order."
)]
fn reference(
    arithmetic: Arithmetic,
    geometry: &Geometry,
    layer: &RecurrentAttention,
    mixed: &[f32],
    gate: &[f32],
    previous: &[f32],
) -> Result<(Vec<f32>, Vec<f32>)> {
    let width = geometry.ssm_head;
    let key_width = geometry.key_width();
    let mut mixed = mixed.to_vec();
    for value in &mut mixed {
        *value = arithmetic.convolution_row(&[0.25; 3], *value, &[0.125, -0.25, 0.5, 0.75])?;
    }
    arithmetic.silu(&mut mixed);
    for head in mixed[..2 * key_width].chunks_exact_mut(width) {
        l2_norm(head, geometry.epsilon);
    }
    let query = &mixed[..width];
    let key = &mixed[key_width..key_width + width];
    let scale = 1.0 / small_float(width)?.sqrt();
    let mut matrix = vec![f32::NAN; previous.len()];
    let mut output = vec![f32::NAN; geometry.value_width()];
    for head in 0..geometry.ssm_values {
        let softplus = (1.0 + layer.time_bias[head].exp()).ln();
        let decay = (softplus * layer.decay[head]).exp();
        let update = sigmoid(0.0);
        let out = &mut output[head * width..(head + 1) * width];
        for (row, value) in out.iter_mut().enumerate() {
            let offset = (head * width + row) * width;
            *value = arithmetic.gdn_step_row_from(
                &previous[offset..offset + width],
                &mut matrix[offset..offset + width],
                key,
                query,
                decay,
                mixed[2 * key_width + head * width + row],
                update,
            )? * scale;
        }
        // Defensive admission is intentionally repeated for each oracle head.
        let gate = &gate[head * width..(head + 1) * width];
        match arithmetic {
            Arithmetic::Source => kernels::rms_norm_swiglu_in_place::<false>(
                out,
                &layer.norm,
                geometry.epsilon,
                gate,
            )?,
            Arithmetic::PinnedSwift => {
                kernels::rms_norm_swiglu_in_place::<true>(
                    out,
                    &layer.norm,
                    geometry.epsilon,
                    gate,
                )?;
            }
        }
    }
    Ok((matrix, output))
}

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "Keep the recurrent-layer fixture, independent row oracle and failed-delta recovery together."
)]
fn recurrent_rms_weight_scope_integrates_worker_heads_and_recovers_failed_layer_delta() -> Result<()>
{
    let width = 128;
    let geometry = Geometry {
        embedding: width,
        feed_forward: width,
        heads: 1,
        kv_heads: 1,
        head: width,
        rope: width,
        rope_sections: [width / 2, 0, 0, 0],
        rope_base: 10_000.0,
        epsilon: 0.000_001,
        ssm_head: width,
        ssm_keys: 1,
        ssm_values: 3,
        convolution: 4,
    };
    let input: Vec<_> = (0..geometry.convolution_width())
        .map(|index| [0.125, -0.25, 0.5, -0.0, f32::from_bits(1)][index % 5])
        .collect();
    let gate = vec![0.75; geometry.value_width()];
    let previous: Vec<_> = (0..width * geometry.value_width())
        .map(|index| [0.125, -0.25, -0.0, f32::from_bits(1)][index % 4])
        .collect();
    let history = vec![0.25; input.len() * 3];
    let source_bits = bits(&previous);
    let worker_counts: &[usize] = if cfg!(feature = "parallel-rows") {
        &[1, 4]
    } else {
        &[1]
    };
    for arithmetic in [Arithmetic::Source, Arithmetic::PinnedSwift] {
        let mut layer = RecurrentAttention {
            qkv: placeholder(),
            gate: placeholder(),
            beta: placeholder(),
            alpha: placeholder(),
            output: placeholder(),
            convolution: kernels::pinned_convolution::Coefficients::new(
                [0.125, -0.25, 0.5, 0.75].repeat(input.len()),
                4,
                matches!(arithmetic, Arithmetic::PinnedSwift),
            ),
            time_bias: vec![0.25; geometry.ssm_values],
            decay: vec![-0.5; geometry.ssm_values],
            norm: vec![1.25; width],
        };
        let (expected_matrix, expected_output) =
            reference(arithmetic, &geometry, &layer, &input, &gate, &previous)?;
        for &workers in worker_counts {
            let mut projection = ProjectionScratch::default();
            projection
                .matvec
                .set_row_parallelism(kernels::RowParallelism::new(workers)?);
            let transforms = Transforms::default();
            let mut adapter = crate::model::lora::Scratch::default();
            let evaluation = Evaluation {
                geometry: &geometry,
                arithmetic,
                transforms: &transforms,
                position: 0,
                frequencies_ready: false,
                projection: &mut projection,
                adapter: None,
                adapter_scale: 0.0,
                adapter_scratch: &mut adapter,
                executor: None,
            };
            let mut delta = LayerDelta::Recurrent {
                convolution: vec![f32::NAN; history.len()],
                matrix: vec![f32::NAN; previous.len()],
            };
            let mut attended = vec![f32::NAN; geometry.value_width()];
            for failure in [0, 1, 2] {
                let mut mixed = input.clone();
                let mut beta = vec![0.0; geometry.ssm_values];
                if failure == 0 {
                    layer.norm[width - 1] = f32::NAN;
                } else if failure == 1 {
                    beta[geometry.ssm_values - 1] = f32::NAN;
                }
                let result = evaluation.recurrent_attention_state(
                    &layer,
                    &mut mixed,
                    &gate,
                    &beta,
                    &vec![0.0; geometry.ssm_values],
                    &history,
                    &previous,
                    &mut delta,
                    &mut attended,
                );
                if failure < 2 {
                    assert!(result.is_err());
                    layer.norm[width - 1] = 1.25;
                } else {
                    result?;
                }
                assert_eq!(bits(&previous), source_bits);
                assert!(
                    history
                        .iter()
                        .all(|value| value.to_bits() == 0.25_f32.to_bits())
                );
            }
            let LayerDelta::Recurrent { matrix, .. } = &delta else {
                return Err(crate::invalid("wrong recurrent delta fixture"));
            };
            assert_eq!(bits(matrix), bits(&expected_matrix));
            assert_eq!(bits(&attended), bits(&expected_output));
        }
    }
    Ok(())
}
