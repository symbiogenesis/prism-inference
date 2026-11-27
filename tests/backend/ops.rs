//! Focused CPU graph storage, aliasing, shape, and numerical regressions.

use bonsai_prism_runtime::Result;
use bonsai_prism_runtime::backend::{Argument, CpuBackend, Observation, TensorId};

fn result_tensor(value: &Observation) -> Result<TensorId> {
    match value {
        Observation::Tensor(Some(id)) => Ok(*id),
        _ => Err(std::io::Error::other("expected tensor result").into()),
    }
}

fn evaluate(backend: &mut CpuBackend, tensor: TensorId) -> Result<()> {
    let Observation::Graph(graph) = backend.call("ggml_new_graph", &[Argument::Context(1)])? else {
        return Err(std::io::Error::other("expected graph result").into());
    };
    backend.call(
        "ggml_build_forward_expand",
        &[Argument::Graph(graph), Argument::Tensor(tensor)],
    )?;
    backend.compute(graph)
}

#[test]
fn glu_odd_rows_preserve_floor_split_swapping_and_separate_inputs() -> Result<()> {
    for kind in [0, 1] {
        let mut backend = CpuBackend::new();
        let a = backend.tensor(1, kind, [5, 2, 1, 1])?;
        backend.set_values(a, &[1.0, -2.0, 3.0, 4.0, 99.0, -5.0, 6.0, 7.0, -8.0, 88.0])?;
        for (swapped, expected) in [
            (false, [3.0, 0.0, 0.0, -48.0]),
            (true, [3.0, -8.0, -35.0, 0.0]),
        ] {
            let glu = result_tensor(&backend.call(
                "ggml_glu",
                &[
                    Argument::Context(1),
                    Argument::Tensor(a),
                    Argument::Integer(0),
                    Argument::Bool(swapped),
                ],
            )?)?;
            assert_eq!(backend.info(glu)?.shape, [2, 2, 1, 1]);
            evaluate(&mut backend, glu)?;
            assert_eq!(backend.values(glu)?, expected);
        }
        let b = backend.tensor(1, kind, [5, 2, 1, 1])?;
        backend.set_values(b, &[3.0, 4.0, -1.0, 2.0, 0.0, 3.0, -1.0, 2.0, 4.0, 0.0])?;
        let glu = result_tensor(&backend.call(
            "ggml_glu_split",
            &[
                Argument::Context(1),
                Argument::Tensor(a),
                Argument::Tensor(b),
                Argument::Integer(0),
            ],
        )?)?;
        assert_eq!(backend.info(glu)?.shape, [5, 2, 1, 1]);
        evaluate(&mut backend, glu)?;
        assert_eq!(
            backend.values(glu)?,
            [3.0, 0.0, -3.0, 8.0, 0.0, 0.0, -6.0, 14.0, 0.0, 0.0]
        );
    }
    Ok(())
}

#[test]
fn strided_transpose_and_broadcast_preserve_logical_values() -> Result<()> {
    let mut backend = CpuBackend::new();
    let a = backend.tensor(1, 0, [2, 3, 1, 1])?;
    backend.set_values(a, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;
    let transposed = result_tensor(&backend.call(
        "ggml_transpose",
        &[Argument::Context(1), Argument::Tensor(a)],
    )?)?;
    assert_eq!(backend.values(transposed)?, [1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);
    let b = backend.tensor(1, 0, [3, 1, 1, 1])?;
    backend.set_values(b, &[10.0, 20.0, 30.0])?;
    let sum = result_tensor(&backend.call(
        "ggml_add",
        &[
            Argument::Context(1),
            Argument::Tensor(transposed),
            Argument::Tensor(b),
        ],
    )?)?;
    evaluate(&mut backend, sum)?;
    assert_eq!(backend.values(sum)?, [11.0, 23.0, 35.0, 12.0, 24.0, 36.0]);
    assert_eq!(backend.values(a)?, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    Ok(())
}

#[test]
fn matmul_and_softmax_have_independent_expected_results() -> Result<()> {
    let mut backend = CpuBackend::new();
    let a = backend.tensor(1, 0, [3, 2, 1, 1])?;
    let b = backend.tensor(1, 0, [3, 2, 1, 1])?;
    backend.set_values(a, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;
    backend.set_values(b, &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0])?;
    let product = result_tensor(&backend.call(
        "ggml_mul_mat",
        &[
            Argument::Context(1),
            Argument::Tensor(a),
            Argument::Tensor(b),
        ],
    )?)?;
    evaluate(&mut backend, product)?;
    assert_eq!(backend.values(product)?, [50.0, 122.0, 68.0, 167.0]);
    let logits = backend.tensor(1, 0, [3, 1, 1, 1])?;
    backend.set_values(logits, &[0.0, 2.0_f32.ln(), 3.0_f32.ln()])?;
    let probabilities = result_tensor(&backend.call(
        "ggml_soft_max",
        &[Argument::Context(1), Argument::Tensor(logits)],
    )?)?;
    evaluate(&mut backend, probabilities)?;
    for (&actual, expected) in
        backend
            .values(probabilities)?
            .iter()
            .zip([1.0 / 6.0, 2.0 / 6.0, 0.5])
    {
        assert!((actual - expected).abs() < 1e-6);
    }
    Ok(())
}

#[test]
fn inplace_views_write_underlying_storage_and_reject_invalid_extent() -> Result<()> {
    let mut backend = CpuBackend::new();
    let a = backend.tensor(1, 0, [4, 1, 1, 1])?;
    backend.set_values(a, &[1.0, 2.0, 3.0, 4.0])?;
    let view = result_tensor(&backend.call(
        "ggml_view_1d",
        &[
            Argument::Context(1),
            Argument::Tensor(a),
            Argument::Integer(2),
            Argument::Integer(4),
        ],
    )?)?;
    let changed = result_tensor(&backend.call(
        "ggml_scale_bias_inplace",
        &[
            Argument::Context(1),
            Argument::Tensor(view),
            Argument::Float(2.0),
            Argument::Float(1.0),
        ],
    )?)?;
    evaluate(&mut backend, changed)?;
    assert_eq!(backend.values(a)?, [1.0, 5.0, 7.0, 4.0]);
    assert!(backend.set(view, 8, &[1]).is_err());
    assert!(
        backend
            .call(
                "ggml_mul_mat",
                &[
                    Argument::Context(1),
                    Argument::Tensor(a),
                    Argument::Tensor(view)
                ]
            )
            .is_err()
    );
    Ok(())
}

#[test]
fn unsupported_operation_never_reports_support_or_computes() -> Result<()> {
    let mut backend = CpuBackend::new();
    let tensor = backend.tensor(1, 0, [1; 4])?;
    backend.info_mut(tensor)?.operation = 75;
    assert!(!backend.supports(tensor)?);
    assert!(evaluate(&mut backend, tensor).is_err());
    assert!(
        backend
            .call(
                "ggml_build_backward_expand",
                &[Argument::Context(1), Argument::Tensor(tensor)]
            )
            .is_err()
    );
    Ok(())
}

#[test]
fn image_extraction_convolution_and_pooling_preserve_axis_order() -> Result<()> {
    let mut backend = CpuBackend::new();
    let image = backend.tensor(1, 0, [3, 3, 1, 1])?;
    backend.set_values(image, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])?;
    let kernel = backend.tensor(1, 0, [2, 2, 1, 1])?;
    backend.set_values(kernel, &[1.0; 4])?;
    let mut arguments = vec![
        Argument::Context(1),
        Argument::Tensor(kernel),
        Argument::Tensor(image),
    ];
    arguments.extend([1, 1, 0, 0, 1, 1].map(Argument::Integer));
    let convolution = result_tensor(&backend.call("ggml_conv_2d_direct", &arguments)?)?;
    evaluate(&mut backend, convolution)?;
    assert_eq!(backend.values(convolution)?, [12.0, 16.0, 24.0, 28.0]);
    arguments.extend([Argument::Bool(true), Argument::Integer(0)]);
    let columns = result_tensor(&backend.call("ggml_im2col", &arguments)?)?;
    evaluate(&mut backend, columns)?;
    assert_eq!(
        backend.values(columns)?,
        [
            1.0, 2.0, 4.0, 5.0, 2.0, 3.0, 5.0, 6.0, 4.0, 5.0, 7.0, 8.0, 5.0, 6.0, 8.0, 9.0
        ]
    );
    let maximum = result_tensor(&backend.call(
        "ggml_pool_2d",
        &[
            Argument::Context(1),
            Argument::Tensor(image),
            Argument::Integer(0),
            Argument::Integer(2),
            Argument::Integer(2),
            Argument::Integer(1),
            Argument::Integer(1),
            Argument::Float(0.0),
            Argument::Float(0.0),
        ],
    )?)?;
    evaluate(&mut backend, maximum)?;
    assert_eq!(backend.values(maximum)?, [5.0, 6.0, 8.0, 9.0]);
    Ok(())
}

#[test]
fn interpolation_and_reflection_use_pinned_pixel_positions() -> Result<()> {
    let mut backend = CpuBackend::new();
    let image = backend.tensor(1, 0, [2, 2, 1, 1])?;
    backend.set_values(image, &[1.0, 2.0, 3.0, 4.0])?;
    let enlarged = result_tensor(&backend.call(
        "ggml_upscale",
        &[
            Argument::Context(1),
            Argument::Tensor(image),
            Argument::Integer(2),
            Argument::Integer(1),
        ],
    )?)?;
    evaluate(&mut backend, enlarged)?;
    assert_eq!(
        backend.values(enlarged)?,
        [
            1.0, 1.25, 1.75, 2.0, 1.5, 1.75, 2.25, 2.5, 2.5, 2.75, 3.25, 3.5, 3.0, 3.25, 3.75, 4.0
        ]
    );
    let constant = backend.tensor(1, 0, [4, 4, 1, 1])?;
    backend.set_values(constant, &[1.0; 16])?;
    for mode in [2, 513] {
        let interpolated = result_tensor(&backend.call(
            "ggml_interpolate",
            &[
                Argument::Context(1),
                Argument::Tensor(constant),
                Argument::Integer(2),
                Argument::Integer(2),
                Argument::Integer(1),
                Argument::Integer(1),
                Argument::Integer(mode),
            ],
        )?)?;
        evaluate(&mut backend, interpolated)?;
        assert!(
            backend
                .values(interpolated)?
                .iter()
                .all(|value| (*value - 1.0).abs() < 1e-6)
        );
    }
    let line = backend.tensor(1, 0, [3, 1, 1, 1])?;
    backend.set_values(line, &[1.0, 2.0, 3.0])?;
    let reflected = result_tensor(&backend.call(
        "ggml_pad_reflect_1d",
        &[
            Argument::Context(1),
            Argument::Tensor(line),
            Argument::Integer(2),
            Argument::Integer(1),
        ],
    )?)?;
    evaluate(&mut backend, reflected)?;
    assert_eq!(backend.values(reflected)?, [3.0, 2.0, 1.0, 2.0, 3.0, 2.0]);
    Ok(())
}

#[test]
fn integer_copies_and_comparisons_preserve_full_i64_domain() -> Result<()> {
    let mut backend = CpuBackend::new();
    let original = backend.tensor(1, 27, [2, 2, 1, 1])?;
    backend.set(
        original,
        0,
        &[i64::MAX, i64::MAX - 1, i64::MIN, i64::MIN + 1]
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>(),
    )?;
    let transposed = result_tensor(&backend.call(
        "ggml_transpose",
        &[Argument::Context(1), Argument::Tensor(original)],
    )?)?;
    let copied = result_tensor(&backend.call(
        "ggml_cont",
        &[Argument::Context(1), Argument::Tensor(transposed)],
    )?)?;
    evaluate(&mut backend, copied)?;
    assert_eq!(
        backend.integer_values(copied)?,
        [i64::MAX, i64::MIN, i64::MAX - 1, i64::MIN + 1]
    );
    let other = backend.tensor(1, 27, [2, 2, 1, 1])?;
    backend.set(
        other,
        0,
        &[i64::MAX - 1, i64::MAX - 1, i64::MIN + 1, i64::MIN + 1]
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>(),
    )?;
    let equal = result_tensor(&backend.call(
        "ggml_count_equal",
        &[
            Argument::Context(1),
            Argument::Tensor(original),
            Argument::Tensor(other),
        ],
    )?)?;
    evaluate(&mut backend, equal)?;
    assert_eq!(backend.integer_values(equal)?, [2]);
    Ok(())
}

#[test]
fn expert_matmul_uses_per_token_ids_and_broadcast_activation_slots() -> Result<()> {
    let mut backend = CpuBackend::new();
    let weights = backend.tensor(1, 0, [2, 2, 2, 1])?;
    backend.set_values(weights, &[1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0])?;
    let activations = backend.tensor(1, 0, [2, 1, 2, 1])?;
    backend.set_values(activations, &[3.0, 4.0, 5.0, 6.0])?;
    let experts = backend.tensor(1, 26, [2, 2, 1, 1])?;
    backend.set_values(experts, &[1.0, 0.0, 0.0, 1.0])?;
    let product = result_tensor(&backend.call(
        "ggml_mul_mat_id",
        &[
            Argument::Context(1),
            Argument::Tensor(weights),
            Argument::Tensor(activations),
            Argument::Tensor(experts),
        ],
    )?)?;
    evaluate(&mut backend, product)?;
    assert_eq!(
        backend.values(product)?,
        [6.0, 8.0, 3.0, 4.0, 5.0, 6.0, 10.0, 12.0]
    );
    let bias = backend.tensor(1, 0, [2, 2, 1, 1])?;
    backend.set_values(bias, &[1.0, 2.0, 3.0, 4.0])?;
    let shifted = result_tensor(&backend.call(
        "ggml_add_id",
        &[
            Argument::Context(1),
            Argument::Tensor(product),
            Argument::Tensor(bias),
            Argument::Tensor(experts),
        ],
    )?)?;
    evaluate(&mut backend, shifted)?;
    assert_eq!(
        backend.values(shifted)?,
        [9.0, 12.0, 4.0, 6.0, 6.0, 8.0, 13.0, 16.0]
    );
    Ok(())
}

#[test]
fn ssm_convolution_and_optimizer_update_persistent_moments() -> Result<()> {
    let mut backend = CpuBackend::new();
    let signal = backend.tensor(1, 0, [4, 2, 1, 1])?;
    backend.set_values(signal, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])?;
    let weights = backend.tensor(1, 0, [2, 2, 1, 1])?;
    backend.set_values(weights, &[1.0, 1.0, 1.0, -1.0])?;
    let filtered = result_tensor(&backend.call(
        "ggml_ssm_conv",
        &[
            Argument::Context(1),
            Argument::Tensor(signal),
            Argument::Tensor(weights),
        ],
    )?)?;
    evaluate(&mut backend, filtered)?;
    assert_eq!(backend.values(filtered)?, [3.0, -1.0, 5.0, -1.0, 7.0, -1.0]);
    let parameter = backend.tensor(2, 0, [2, 1, 1, 1])?;
    backend.set_values(parameter, &[3.0, 5.0])?;
    backend.call("ggml_set_param", &[Argument::Tensor(parameter)])?;
    let gradient = backend.tensor(2, 0, [2, 1, 1, 1])?;
    backend.set_values(gradient, &[2.0, -4.0])?;
    let moment = backend.tensor(2, 0, [2, 1, 1, 1])?;
    backend.set_values(moment, &[0.0, 0.0])?;
    let variance = backend.tensor(2, 0, [2, 1, 1, 1])?;
    backend.set_values(variance, &[0.0, 0.0])?;
    let options = backend.tensor(2, 0, [7, 1, 1, 1])?;
    backend.set_values(options, &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0])?;
    let updated = result_tensor(&backend.call(
        "ggml_opt_step_adamw",
        &[
            Argument::Context(2),
            Argument::Tensor(parameter),
            Argument::Tensor(gradient),
            Argument::Tensor(moment),
            Argument::Tensor(variance),
            Argument::Tensor(options),
        ],
    )?)?;
    evaluate(&mut backend, updated)?;
    assert_eq!(backend.values(parameter)?, [2.0, 6.0]);
    assert_eq!(backend.values(moment)?, [2.0, -4.0]);
    assert_eq!(backend.values(variance)?, [4.0, 16.0]);
    Ok(())
}

#[test]
fn attention_and_softmax_sinks_preserve_absent_mask_slots() -> Result<()> {
    let mut backend = CpuBackend::new();
    let queries = backend.tensor(1, 0, [2, 1, 2, 1])?;
    backend.set_values(queries, &[0.0; 4])?;
    let keys = backend.tensor(1, 1, [2, 2, 1, 1])?;
    backend.set_values(keys, &[1.0; 4])?;
    let values = backend.tensor(1, 1, [1, 2, 1, 1])?;
    backend.set_values(values, &[1.0, 3.0])?;
    let sinks = backend.tensor(1, 0, [2, 1, 1, 1])?;
    backend.set_values(sinks, &[0.0, 2.0_f32.ln()])?;
    let attention = result_tensor(&backend.call(
        "ggml_flash_attn_ext",
        &[
            Argument::Context(1),
            Argument::Tensor(queries),
            Argument::Tensor(keys),
            Argument::Tensor(values),
            Argument::Null,
            Argument::Float(1.0),
            Argument::Float(0.0),
            Argument::Float(0.0),
        ],
    )?)?;
    backend.call(
        "ggml_flash_attn_ext_add_sinks",
        &[Argument::Tensor(attention), Argument::Tensor(sinks)],
    )?;
    assert!(backend.info(attention)?.source_slots[3].is_none());
    assert_eq!(backend.info(attention)?.source_slots[4], Some(sinks));
    evaluate(&mut backend, attention)?;
    for (&actual, expected) in backend.values(attention)?.iter().zip([4.0 / 3.0, 1.0]) {
        assert!((actual - expected).abs() < 1e-6);
    }
    let probabilities = result_tensor(&backend.call(
        "ggml_soft_max",
        &[Argument::Context(1), Argument::Tensor(queries)],
    )?)?;
    backend.call(
        "ggml_soft_max_add_sinks",
        &[Argument::Tensor(probabilities), Argument::Tensor(sinks)],
    )?;
    assert!(backend.info(probabilities)?.source_slots[1].is_none());
    assert_eq!(backend.info(probabilities)?.source_slots[2], Some(sinks));
    evaluate(&mut backend, probabilities)?;
    for (&actual, expected) in
        backend
            .values(probabilities)?
            .iter()
            .zip([1.0 / 3.0, 1.0 / 3.0, 0.25, 0.25])
    {
        assert!((actual - expected).abs() < 1e-6);
    }
    Ok(())
}

#[test]
fn recurrent_attention_retains_exact_sequence_state_updates() -> Result<()> {
    let mut backend = CpuBackend::new();
    let mut vector = |values: &[f32]| -> Result<TensorId> {
        let id = backend.tensor(1, 0, [1, 1, values.len(), 1])?;
        backend.set_values(id, values)?;
        Ok(id)
    };
    let keys = vector(&[1.0, 2.0])?;
    let values = vector(&[3.0, 4.0])?;
    let query = vector(&[1.0, 1.0])?;
    let gates = vector(&[0.25, 0.5])?;
    let coefficient = vector(&[0.5, 0.5])?;
    let state = backend.tensor(1, 0, [1, 1, 1, 1])?;
    backend.set_values(state, &[2.0])?;
    let time_weight = backend.tensor(1, 0, [1, 1, 1, 1])?;
    backend.set_values(time_weight, &[0.5])?;
    let wkv = result_tensor(&backend.call(
        "ggml_rwkv_wkv6",
        &[
            Argument::Context(1),
            Argument::Tensor(keys),
            Argument::Tensor(values),
            Argument::Tensor(query),
            Argument::Tensor(time_weight),
            Argument::Tensor(gates),
            Argument::Tensor(state),
        ],
    )?)?;
    evaluate(&mut backend, wkv)?;
    assert_eq!(backend.values(wkv)?, [3.5, 7.5, 9.75]);
    let linear = result_tensor(&backend.call(
        "ggml_gated_linear_attn",
        &[
            Argument::Context(1),
            Argument::Tensor(keys),
            Argument::Tensor(values),
            Argument::Tensor(query),
            Argument::Tensor(gates),
            Argument::Tensor(state),
            Argument::Float(2.0),
        ],
    )?)?;
    evaluate(&mut backend, linear)?;
    assert_eq!(backend.values(linear)?, [7.0, 19.5, 9.75]);
    let newer = result_tensor(&backend.call(
        "ggml_rwkv_wkv7",
        &[
            Argument::Context(1),
            Argument::Tensor(query),
            Argument::Tensor(gates),
            Argument::Tensor(keys),
            Argument::Tensor(values),
            Argument::Tensor(coefficient),
            Argument::Tensor(coefficient),
            Argument::Tensor(state),
        ],
    )?)?;
    evaluate(&mut backend, newer)?;
    assert_eq!(backend.values(newer)?, [4.0, 11.0, 11.0]);
    assert_eq!(backend.values(state)?, [2.0]);
    Ok(())
}

#[test]
fn state_scan_keeps_reverse_chronological_snapshot_slots() -> Result<()> {
    let mut backend = CpuBackend::new();
    let state = backend.tensor(1, 0, [1, 1, 1, 1])?;
    backend.set_values(state, &[2.0])?;
    let input = backend.tensor(1, 0, [1, 1, 2, 1])?;
    backend.set_values(input, &[1.0, 2.0])?;
    let delta = backend.tensor(1, 0, [1, 2, 1, 1])?;
    backend.set_values(delta, &[0.0, 0.0])?;
    let decay = backend.tensor(1, 0, [1, 1, 1, 1])?;
    backend.set_values(decay, &[0.0])?;
    let projection = backend.tensor(1, 0, [1, 1, 2, 1])?;
    backend.set_values(projection, &[1.0, 1.0])?;
    let rows = backend.tensor(1, 26, [1, 1, 1, 1])?;
    backend.set_values(rows, &[0.0])?;
    let scan = result_tensor(&backend.call(
        "ggml_ssm_scan",
        &[
            Argument::Context(1),
            Argument::Tensor(state),
            Argument::Tensor(input),
            Argument::Tensor(delta),
            Argument::Tensor(decay),
            Argument::Tensor(projection),
            Argument::Tensor(projection),
            Argument::Tensor(rows),
            Argument::Integer(2),
        ],
    )?)?;
    evaluate(&mut backend, scan)?;
    let first = 2.0 + 2.0_f32.ln();
    let second = 3.0_f32.mul_add(2.0_f32.ln(), 2.0);
    for (&actual, expected) in backend
        .values(scan)?
        .iter()
        .zip([first, second, second, first])
    {
        assert!((actual - expected).abs() < 1e-6);
    }
    assert_eq!(backend.values(state)?, [2.0]);
    Ok(())
}

#[test]
fn hyperconnections_preserve_channel_matrix_orientation() -> Result<()> {
    let mut backend = CpuBackend::new();
    let residual = backend.tensor(1, 0, [2, 2, 1, 1])?;
    backend.set_values(residual, &[1.0, 2.0, 3.0, 4.0])?;
    let weights = backend.tensor(1, 0, [2, 1, 1, 1])?;
    backend.set_values(weights, &[0.25, 0.75])?;
    let reduced = result_tensor(&backend.call(
        "ggml_dsv4_hc_pre",
        &[
            Argument::Context(1),
            Argument::Tensor(residual),
            Argument::Tensor(weights),
        ],
    )?)?;
    evaluate(&mut backend, reduced)?;
    assert_eq!(backend.values(reduced)?, [2.5, 3.5]);
    let signal = backend.tensor(1, 0, [2, 1, 1, 1])?;
    backend.set_values(signal, &[10.0, 20.0])?;
    let post = backend.tensor(1, 0, [2, 1, 1, 1])?;
    backend.set_values(post, &[2.0, 3.0])?;
    let combination = backend.tensor(1, 0, [2, 2, 1, 1])?;
    backend.set_values(combination, &[1.0, 0.0, 0.0, 1.0])?;
    let expanded = result_tensor(&backend.call(
        "ggml_dsv4_hc_post",
        &[
            Argument::Context(1),
            Argument::Tensor(signal),
            Argument::Tensor(residual),
            Argument::Tensor(post),
            Argument::Tensor(combination),
        ],
    )?)?;
    evaluate(&mut backend, expanded)?;
    assert_eq!(backend.values(expanded)?, [21.0, 42.0, 33.0, 64.0]);
    let mixes = backend.tensor(1, 0, [24, 1, 1, 1])?;
    backend.set_values(mixes, &[0.0; 24])?;
    let scale = backend.tensor(1, 0, [3, 1, 1, 1])?;
    backend.set_values(scale, &[1.0; 3])?;
    let balanced = result_tensor(&backend.call(
        "ggml_dsv4_hc_comb",
        &[
            Argument::Context(1),
            Argument::Tensor(mixes),
            Argument::Tensor(scale),
            Argument::Tensor(mixes),
            Argument::Float(0.0),
            Argument::Integer(2),
        ],
    )?)?;
    evaluate(&mut backend, balanced)?;
    assert_eq!(backend.values(balanced)?, [0.25; 16]);
    Ok(())
}

#[test]
fn delta_attention_snapshots_and_cache_rows_preserve_initial_states() -> Result<()> {
    let mut backend = CpuBackend::new();
    let mut vector = |values: &[f32]| -> Result<TensorId> {
        let id = backend.tensor(1, 0, [1, 1, values.len(), 1])?;
        backend.set_values(id, values)?;
        Ok(id)
    };
    let query = vector(&[1.0, 1.0])?;
    let values = vector(&[3.0, 4.0])?;
    let gate = vector(&[0.0, 0.0])?;
    let beta = vector(&[0.5, 0.5])?;
    let state = backend.tensor(1, 0, [1; 4])?;
    backend.set_values(state, &[2.0])?;
    let mut arguments = vec![
        Argument::Context(1),
        Argument::Tensor(query),
        Argument::Tensor(query),
        Argument::Tensor(values),
        Argument::Tensor(gate),
        Argument::Tensor(beta),
        Argument::Tensor(state),
        Argument::Integer(2),
    ];
    let result = result_tensor(&backend.call("ggml_gated_delta_net", &arguments)?)?;
    evaluate(&mut backend, result)?;
    assert_eq!(backend.values(result)?, [2.5, 3.25, 3.25, 2.5]);
    let cache = backend.tensor(1, 0, [1, 3, 1, 1])?;
    backend.set_values(cache, &[2.0, 5.0, 7.0])?;
    let rows = backend.tensor(1, 26, [1; 4])?;
    backend.set_values(rows, &[1.0])?;
    arguments[6] = Argument::Tensor(cache);
    arguments[7] = Argument::Tensor(rows);
    arguments.push(Argument::Integer(2));
    let gathered = result_tensor(&backend.call("ggml_gated_delta_net_rows", &arguments)?)?;
    evaluate(&mut backend, gathered)?;
    assert_eq!(backend.values(gathered)?, [4.0, 4.0, 4.0, 4.0]);
    assert_eq!(backend.values(cache)?, [2.0, 5.0, 7.0]);
    Ok(())
}

#[test]
fn delta_replay_applies_only_selected_bank_and_retained_records() -> Result<()> {
    let mut backend = CpuBackend::new();
    let mut vector = |values: &[f32]| -> Result<TensorId> {
        let id = backend.tensor(1, 0, [1, 1, values.len(), 1])?;
        backend.set_values(id, values)?;
        Ok(id)
    };
    let query = vector(&[1.0, 1.0])?;
    let values = vector(&[3.0, 4.0])?;
    let zero = vector(&[0.0, 0.0])?;
    let state = backend.tensor(1, 0, [1; 4])?;
    backend.set_values(state, &[3.0])?;
    let records = backend.tensor(1, 0, [6, 2, 1, 1])?;
    backend.set_values(
        records,
        &[1.0, 1.0, 1.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0],
    )?;
    let control = backend.tensor(1, 26, [2, 1, 1, 1])?;
    backend.set_values(control, &[1.0, 0.0])?;
    let replay = result_tensor(&backend.call(
        "ggml_gated_delta_net_replay",
        &[
            Argument::Context(1),
            Argument::Tensor(query),
            Argument::Tensor(query),
            Argument::Tensor(values),
            Argument::Tensor(zero),
            Argument::Tensor(zero),
            Argument::Tensor(state),
            Argument::Tensor(records),
            Argument::Tensor(control),
            Argument::Integer(3),
        ],
    )?)?;
    evaluate(&mut backend, replay)?;
    assert_eq!(
        backend.values(replay)?,
        [4.0, 4.0, 4.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    );
    assert_eq!(backend.values(state)?, [3.0]);
    Ok(())
}

#[test]
fn raw_delta_gates_preserve_nullable_row_slot_and_apply_activations() -> Result<()> {
    let mut backend = CpuBackend::new();
    let query = backend.tensor(1, 0, [2, 1, 1, 1])?;
    backend.set_values(query, &[1.0, 0.0])?;
    let values = backend.tensor(1, 0, [2, 1, 1, 1])?;
    backend.set_values(values, &[2.0, 4.0])?;
    let zero = backend.tensor(1, 0, [1; 4])?;
    backend.set_values(zero, &[0.0])?;
    let state = backend.tensor(1, 0, [2, 2, 1, 1])?;
    backend.set_values(state, &[0.0; 4])?;
    let delta = result_tensor(&backend.call(
        "ggml_gated_delta_net",
        &[
            Argument::Context(1),
            Argument::Tensor(query),
            Argument::Tensor(query),
            Argument::Tensor(values),
            Argument::Tensor(zero),
            Argument::Tensor(zero),
            Argument::Tensor(state),
            Argument::Integer(1),
        ],
    )?)?;
    backend.call(
        "ggml_gated_delta_net_set_raw_gates",
        &[
            Argument::Tensor(delta),
            Argument::Tensor(zero),
            Argument::Tensor(zero),
        ],
    )?;
    assert!(backend.info(delta)?.source_slots[6].is_none());
    assert_eq!(backend.info(delta)?.source_slots[7], Some(zero));
    evaluate(&mut backend, delta)?;
    let actual = backend.values(delta)?;
    assert!((actual[0] - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-6);
    assert!((actual[1] - std::f32::consts::SQRT_2).abs() < 1e-6);
    assert_eq!(&actual[2..], &[1.0, 0.0, 2.0, 0.0]);
    Ok(())
}

#[test]
fn lightning_indexer_combines_relu_heads_and_half_mask() -> Result<()> {
    let mut backend = CpuBackend::new();
    let queries = backend.tensor(1, 0, [2, 2, 1, 1])?;
    backend.set_values(queries, &[1.0, 2.0, -1.0, 1.0])?;
    let keys = backend.tensor(1, 1, [2, 1, 2, 1])?;
    backend.set_values(keys, &[2.0, 0.0, 0.0, 3.0])?;
    let weights = backend.tensor(1, 0, [2, 1, 1, 1])?;
    backend.set_values(weights, &[0.5, 2.0])?;
    let mask = backend.tensor(1, 1, [2, 1, 1, 1])?;
    backend.set_values(mask, &[0.0, -1.0])?;
    let scores = result_tensor(&backend.call(
        "ggml_lightning_indexer",
        &[
            Argument::Context(1),
            Argument::Tensor(queries),
            Argument::Tensor(keys),
            Argument::Tensor(weights),
            Argument::Tensor(mask),
        ],
    )?)?;
    evaluate(&mut backend, scores)?;
    assert_eq!(backend.values(scores)?, [1.0, 8.0]);
    Ok(())
}

#[test]
fn volume_extraction_preserves_empty_source_view_geometry_1843() -> Result<()> {
    // Unchanged source geometry 1843: viewing [20,20,10,3] removes two
    // elements from every axis. IC=3 therefore gives zero complete batches.
    let mut backend = CpuBackend::new();
    let input = backend.tensor(1, 0, [20, 20, 10, 3])?;
    let strides = backend.info(input)?.strides;
    let view = result_tensor(&backend.call(
        "ggml_view_4d",
        &[
            Argument::Context(1),
            Argument::Tensor(input),
            Argument::Integer(18),
            Argument::Integer(18),
            Argument::Integer(8),
            Argument::Integer(1),
            Argument::Integer(i64::try_from(strides[1])?),
            Argument::Integer(i64::try_from(strides[2])?),
            Argument::Integer(i64::try_from(strides[3])?),
            Argument::Integer(0),
        ],
    )?)?;
    let kernel = backend.tensor(1, 0, [3, 3, 3, 3])?;
    let mut arguments = vec![
        Argument::Context(1),
        Argument::Tensor(kernel),
        Argument::Tensor(view),
        Argument::Integer(3),
    ];
    arguments.extend([1, 1, 1, 0, 0, 0, 1, 3, 1, 0].map(Argument::Integer));
    let columns = result_tensor(&backend.call("ggml_im2col_3d", &arguments)?)?;
    let info = backend.info(columns)?;
    assert_eq!(info.shape, [81, 16, 12, 0]);
    assert_eq!(info.strides, [4, 324, 5184, 62208]);
    assert_eq!(info.bytes()?, 0);
    assert_eq!(info.elements(), 0);
    assert_eq!(
        info.parameters[..40],
        [1_i32, 1, 1, 0, 0, 0, 1, 3, 1, 3]
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>()
    );
    assert!(backend.supports(columns)?);
    // Inputs deliberately have no allocated payload: an empty native node
    // does not touch them or divide by its zero batch count.
    evaluate(&mut backend, columns)?;
    assert_eq!(backend.values(columns)?, [0.0_f32; 0]);
    arguments[3] = Argument::Integer(0);
    assert!(backend.call("ggml_im2col_3d", &arguments).is_err());
    arguments[3] = Argument::Integer(3);
    arguments[4] = Argument::Integer(0);
    assert!(backend.call("ggml_im2col_3d", &arguments).is_err());
    Ok(())
}

#[test]
fn volume_extraction_uses_complete_batches_and_ignores_trailing_channel_planes() -> Result<()> {
    for output_kind in [0, 1] {
        let mut backend = CpuBackend::new();
        let input = backend.tensor(1, 0, [2, 1, 1, 7])?;
        backend.set_values(
            input,
            &[
                1.0, 2.0, 11.0, 12.0, 21.0, 22.0, 31.0, 32.0, 41.0, 42.0, 51.0, 52.0, 999.0, 999.0,
            ],
        )?;
        let kernel = backend.tensor(1, 0, [1, 1, 1, 3])?;
        backend.set_values(kernel, &[0.0; 3])?;
        let mut arguments = vec![
            Argument::Context(1),
            Argument::Tensor(kernel),
            Argument::Tensor(input),
            Argument::Integer(3),
        ];
        arguments.extend([1, 1, 1, 0, 0, 0, 1, 1, 1, output_kind].map(Argument::Integer));
        let columns = result_tensor(&backend.call("ggml_im2col_3d", &arguments)?)?;
        assert_eq!(backend.info(columns)?.shape, [3, 2, 1, 2]);
        evaluate(&mut backend, columns)?;
        assert_eq!(
            backend.values(columns)?,
            [
                1.0, 11.0, 21.0, 2.0, 12.0, 22.0, 31.0, 41.0, 51.0, 32.0, 42.0, 52.0
            ]
        );
    }
    Ok(())
}

#[test]
fn volume_extraction_and_channel_contiguous_depthwise_keep_strides() -> Result<()> {
    let mut backend = CpuBackend::new();
    let volume = backend.tensor(1, 0, [3, 1, 3, 1])?;
    backend.set_values(volume, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])?;
    let kernel = backend.tensor(1, 0, [2, 1, 2, 1])?;
    backend.set_values(kernel, &[1.0; 4])?;
    let mut arguments = vec![
        Argument::Context(1),
        Argument::Tensor(kernel),
        Argument::Tensor(volume),
        Argument::Integer(1),
    ];
    arguments.extend([1, 1, 1, 0, 0, 0, 1, 1, 1, 0].map(Argument::Integer));
    let columns = result_tensor(&backend.call("ggml_im2col_3d", &arguments)?)?;
    evaluate(&mut backend, columns)?;
    assert_eq!(backend.info(columns)?.shape, [4, 2, 1, 2]);
    assert_eq!(
        backend.values(columns)?,
        [
            1.0, 2.0, 4.0, 5.0, 2.0, 3.0, 5.0, 6.0, 4.0, 5.0, 7.0, 8.0, 5.0, 6.0, 8.0, 9.0
        ]
    );
    let channels = backend.tensor(1, 0, [2, 3, 2, 1])?;
    backend.set_values(
        channels,
        &[1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0],
    )?;
    let permuted = result_tensor(&backend.call(
        "ggml_permute",
        &[
            Argument::Context(1),
            Argument::Tensor(channels),
            Argument::Integer(2),
            Argument::Integer(0),
            Argument::Integer(1),
            Argument::Integer(3),
        ],
    )?)?;
    let channel_kernel = backend.tensor(1, 0, [1, 1, 1, 2])?;
    backend.set_values(channel_kernel, &[3.0, 4.0])?;
    let mut arguments = vec![
        Argument::Context(1),
        Argument::Tensor(channel_kernel),
        Argument::Tensor(permuted),
    ];
    arguments.extend([1, 1, 0, 0, 1, 1].map(Argument::Integer));
    let convolved = result_tensor(&backend.call("ggml_conv_2d_dw_direct", &arguments)?)?;
    evaluate(&mut backend, convolved)?;
    assert_eq!(backend.info(convolved)?.strides, [8, 24, 4, 48]);
    assert_eq!(
        backend.values(convolved)?,
        [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0]
    );
    Ok(())
}
