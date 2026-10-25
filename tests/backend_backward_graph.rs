#![forbid(unsafe_code)]
//! Focused symbolic graph derivatives, accumulation, ownership and rejection checks.

use bonsai_prism_runtime::Result;
use bonsai_prism_runtime::backend::{Argument, CpuBackend, GraphId, Observation, TensorId};

fn operation(backend: &mut CpuBackend, name: &str, arguments: &[Argument]) -> Result<TensorId> {
    let mut args = vec![Argument::Context(1)];
    args.extend_from_slice(arguments);
    let Observation::Tensor(Some(result)) = backend.call(name, &args)? else {
        return Err("operation returned no tensor".into());
    };
    Ok(result)
}

fn unary(backend: &mut CpuBackend, name: &str, a: TensorId) -> Result<TensorId> {
    operation(backend, name, &[Argument::Tensor(a)])
}

fn binary(backend: &mut CpuBackend, name: &str, a: TensorId, b: TensorId) -> Result<TensorId> {
    operation(backend, name, &[Argument::Tensor(a), Argument::Tensor(b)])
}

fn input(
    backend: &mut CpuBackend,
    shape: [usize; 4],
    values: &[f32],
    parameter: bool,
) -> Result<TensorId> {
    let tensor = backend.tensor(1, 0, shape)?;
    backend.set_values(tensor, values)?;
    if parameter {
        backend.call("ggml_set_param", &[Argument::Tensor(tensor)])?;
    }
    Ok(tensor)
}

fn make_graph(backend: &mut CpuBackend, loss: TensorId) -> Result<GraphId> {
    backend.call("ggml_set_loss", &[Argument::Tensor(loss)])?;
    let Observation::Graph(graph) = backend.call(
        "ggml_new_graph_custom",
        &[
            Argument::Context(1),
            Argument::Integer(512),
            Argument::Bool(true),
        ],
    )?
    else {
        return Err("graph result missing".into());
    };
    backend.call(
        "ggml_build_forward_expand",
        &[Argument::Graph(graph), Argument::Tensor(loss)],
    )?;
    Ok(graph)
}

fn gradient(backend: &CpuBackend, graph: GraphId, tensor: TensorId) -> Result<Vec<f32>> {
    backend.values(
        backend
            .graph_gradient(graph, tensor)?
            .ok_or("gradient missing")?,
    )
}

fn close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance,
            "coordinate {index}: {actual} != {expected}"
        );
    }
}

#[test]
fn branch_broadcast_gradients_and_owned_accumulators_reset() -> Result<()> {
    let mut backend = CpuBackend::new();
    let x = input(&mut backend, [2, 2, 1, 1], &[1.0, 2.0, 3.0, 4.0], true)?;
    let bias = input(&mut backend, [2, 1, 1, 1], &[3.0, 4.0], true)?;
    let product = binary(&mut backend, "ggml_mul", x, bias)?;
    let square = unary(&mut backend, "ggml_sqr", x)?;
    let output = binary(&mut backend, "ggml_add", product, square)?;
    let loss = unary(&mut backend, "ggml_sum", output)?;
    let graph = make_graph(&mut backend, loss)?;
    let accumulator = backend.tensor(1, 0, [2, 2, 1, 1])?;
    let (nodes, _) = backend.graph_nodes(graph)?;
    let mut accumulators = vec![None; nodes.len()];
    let index = nodes
        .iter()
        .position(|&tensor| tensor == x)
        .ok_or("parameter missing")?;
    accumulators[index] = Some(accumulator);
    backend.build_backward(1, graph, Some(&accumulators))?;
    assert_eq!(
        backend.graph_gradient_accumulator(graph, x)?,
        Some(accumulator)
    );
    backend.reset_graph(graph)?;
    backend.compute(graph)?;
    close(&gradient(&backend, graph, x)?, &[5.0, 8.0, 9.0, 12.0], 0.0);
    close(&gradient(&backend, graph, bias)?, &[4.0, 6.0], 0.0);
    backend.compute(graph)?;
    close(
        &gradient(&backend, graph, x)?,
        &[10.0, 16.0, 18.0, 24.0],
        0.0,
    );
    backend.reset_graph(graph)?;
    close(&backend.values(accumulator)?, &[0.0; 4], 0.0);
    backend.compute(graph)?;
    close(&gradient(&backend, graph, x)?, &[5.0, 8.0, 9.0, 12.0], 0.0);
    close(&backend.values(x)?, &[1.0, 2.0, 3.0, 4.0], 0.0);
    Ok(())
}

#[test]
fn strided_view_offsets_and_permutation_scatter_into_source_gradients() -> Result<()> {
    let mut backend = CpuBackend::new();
    let x = input(
        &mut backend,
        [4, 2, 1, 1],
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        true,
    )?;
    let view = operation(
        &mut backend,
        "ggml_view_2d",
        &[
            Argument::Tensor(x),
            Argument::Integer(2),
            Argument::Integer(2),
            Argument::Integer(16),
            Argument::Integer(4),
        ],
    )?;
    let transpose = unary(&mut backend, "ggml_transpose", view)?;
    let square = unary(&mut backend, "ggml_sqr", transpose)?;
    let loss = unary(&mut backend, "ggml_sum", square)?;
    let graph = make_graph(&mut backend, loss)?;
    backend.build_backward(1, graph, None)?;
    backend.reset_graph(graph)?;
    backend.compute(graph)?;
    close(
        &gradient(&backend, graph, x)?,
        &[0.0, 4.0, 6.0, 0.0, 0.0, 12.0, 14.0, 0.0],
        0.0,
    );
    let permute = operation(
        &mut backend,
        "ggml_permute",
        &[
            Argument::Tensor(x),
            Argument::Integer(1),
            Argument::Integer(2),
            Argument::Integer(0),
            Argument::Integer(3),
        ],
    )?;
    let square = unary(&mut backend, "ggml_sqr", permute)?;
    let loss = unary(&mut backend, "ggml_sum", square)?;
    let graph = make_graph(&mut backend, loss)?;
    backend.build_backward(1, graph, None)?;
    backend.reset_graph(graph)?;
    backend.compute(graph)?;
    close(
        &gradient(&backend, graph, x)?,
        &[2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0],
        0.0,
    );
    Ok(())
}

#[test]
fn matmul_gradients_use_both_actual_operands() -> Result<()> {
    let mut backend = CpuBackend::new();
    let a = input(
        &mut backend,
        [2, 3, 1, 1],
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        true,
    )?;
    let b = input(&mut backend, [2, 2, 1, 1], &[1.0, 2.0, 3.0, 4.0], true)?;
    let product = binary(&mut backend, "ggml_mul_mat", a, b)?;
    let loss = unary(&mut backend, "ggml_sum", product)?;
    let graph = make_graph(&mut backend, loss)?;
    backend.build_backward(1, graph, None)?;
    backend.reset_graph(graph)?;
    backend.compute(graph)?;
    close(
        &gradient(&backend, graph, a)?,
        &[4.0, 6.0, 4.0, 6.0, 4.0, 6.0],
        0.0,
    );
    close(&gradient(&backend, graph, b)?, &[9.0, 12.0, 9.0, 12.0], 0.0);
    Ok(())
}

#[test]
fn rms_silu_scaled_softmax_chain_matches_finite_differences() -> Result<()> {
    let mut backend = CpuBackend::new();
    let values = [-1.2, 0.4, 1.7, -0.6, 1.1, 0.2];
    let x = input(&mut backend, [3, 2, 1, 1], &values, true)?;
    let weights = input(
        &mut backend,
        [3, 2, 1, 1],
        &[0.2, -0.7, 1.3, 0.9, -0.5, 0.4],
        false,
    )?;
    let rms = operation(
        &mut backend,
        "ggml_rms_norm",
        &[Argument::Tensor(x), Argument::Float(0.01)],
    )?;
    let silu = unary(&mut backend, "ggml_silu", rms)?;
    let softmax = operation(
        &mut backend,
        "ggml_soft_max_ext",
        &[
            Argument::Tensor(silu),
            Argument::Null,
            Argument::Float(2.3),
            Argument::Float(0.0),
        ],
    )?;
    let product = binary(&mut backend, "ggml_mul", softmax, weights)?;
    let loss = unary(&mut backend, "ggml_sum", product)?;
    let forward = make_graph(&mut backend, loss)?;
    let Observation::Graph(backward) = backend.call(
        "ggml_new_graph_custom",
        &[
            Argument::Context(1),
            Argument::Integer(512),
            Argument::Bool(true),
        ],
    )?
    else {
        return Err("graph result missing".into());
    };
    backend.call(
        "ggml_graph_cpy",
        &[Argument::Graph(forward), Argument::Graph(backward)],
    )?;
    backend.build_backward(1, backward, None)?;
    backend.reset_graph(backward)?;
    backend.compute(backward)?;
    let algebraic = gradient(&backend, backward, x)?;
    for index in 0..values.len() {
        let mut changed = values;
        changed[index] += 0.001;
        backend.set_values(x, &changed)?;
        backend.compute(forward)?;
        let plus = backend.values(loss)?[0];
        changed[index] -= 0.002;
        backend.set_values(x, &changed)?;
        backend.compute(forward)?;
        let minus = backend.values(loss)?[0];
        close(&[algebraic[index]], &[(plus - minus) / 0.002], 0.0003);
    }
    Ok(())
}

#[test]
fn cross_entropy_backward_uses_gradient_logits_labels_source_order() -> Result<()> {
    for (shape, repeats, rows) in [
        ([3, 2, 1, 1], 1, 2.0_f32),
        ([3, 2, 2, 1], 2, 4.0),
        ([3, 2, 2, 2], 4, 8.0),
    ] {
        let mut backend = CpuBackend::new();
        let logits_values: Vec<_> = [0.2, -0.4, 0.7, 0.3, 1.0, -0.2]
            .into_iter()
            .cycle()
            .take(6 * repeats)
            .collect();
        let label_values: Vec<_> = [0.0, 1.0, 0.0, 1.0, 0.0, 0.0]
            .into_iter()
            .cycle()
            .take(6 * repeats)
            .collect();
        let logits = input(&mut backend, shape, &logits_values, true)?;
        let labels = input(&mut backend, shape, &label_values, false)?;
        let loss = binary(&mut backend, "ggml_cross_entropy_loss", logits, labels)?;
        let graph = make_graph(&mut backend, loss)?;
        backend.build_backward(1, graph, None)?;
        backend.reset_graph(graph)?;
        backend.compute(graph)?;
        let actual = gradient(&backend, graph, logits)?;
        let mut expected = Vec::new();
        let mut expected_loss = 0.0;
        for (row, targets) in backend
            .values(logits)?
            .as_chunks::<3>()
            .0
            .iter()
            .zip(backend.values(labels)?.as_chunks::<3>().0.iter())
        {
            let total = row.iter().map(|value| value.exp()).sum::<f32>();
            expected_loss -= row
                .iter()
                .zip(targets)
                .map(|(value, target)| target * (value - total.ln()) / rows)
                .sum::<f32>();
            expected.extend(
                row.iter()
                    .zip(targets)
                    .map(|(value, target)| (value.exp() / total - target) / rows),
            );
        }
        close(&actual, &expected, 0.000_001);
        close(&backend.values(loss)?, &[expected_loss], 0.000_001);
        assert_eq!(
            backend
                .info(
                    backend
                        .graph_gradient(graph, logits)?
                        .ok_or("gradient missing")?
                )?
                .sources
                .len(),
            3
        );
    }
    Ok(())
}

#[test]
fn add1_preserves_pinned_mean_derivative_and_piecewise_constant_inputs() -> Result<()> {
    let mut backend = CpuBackend::new();
    let x = input(&mut backend, [4, 1, 1, 1], &[1.0, -2.0, 3.0, 4.0], true)?;
    let scalar = input(&mut backend, [1; 4], &[0.5], true)?;
    let sum = binary(&mut backend, "ggml_add1", x, scalar)?;
    let loss = unary(&mut backend, "ggml_sum", sum)?;
    let graph = make_graph(&mut backend, loss)?;
    backend.build_backward(1, graph, None)?;
    backend.reset_graph(graph)?;
    backend.compute(graph)?;
    // This is intentionally the pinned mean, whose upstream TODO questions
    // replacing it with sum. A port must not silently change that contract.
    close(&gradient(&backend, graph, scalar)?, &[1.0], 0.0);
    let sign = operation(
        &mut backend,
        "ggml_unary",
        &[Argument::Tensor(x), Argument::Integer(1)],
    )?;
    let loss = unary(&mut backend, "ggml_sum", sign)?;
    let graph = make_graph(&mut backend, loss)?;
    backend.build_backward(1, graph, None)?;
    assert_eq!(backend.graph_gradient(graph, x)?, None);
    Ok(())
}

#[test]
fn unsupported_expansion_rolls_back_metadata_and_preserves_valid_graph() -> Result<()> {
    let mut backend = CpuBackend::new();
    let x = input(&mut backend, [2, 1, 1, 1], &[1.0, 2.0], true)?;
    let square = unary(&mut backend, "ggml_sqr", x)?;
    let loss = unary(&mut backend, "ggml_sum", square)?;
    let valid = make_graph(&mut backend, loss)?;
    backend.build_backward(1, valid, None)?;
    let existing_gradient = backend.graph_gradient(valid, x)?;
    let gate = input(&mut backend, [2, 1, 1, 1], &[0.3, 0.8], false)?;
    let geglu = operation(
        &mut backend,
        "ggml_glu_split",
        &[
            Argument::Tensor(x),
            Argument::Tensor(gate),
            Argument::Integer(1),
        ],
    )?;
    let loss = unary(&mut backend, "ggml_sum", geglu)?;
    let invalid = make_graph(&mut backend, loss)?;
    let nodes = backend.graph_nodes(invalid)?.0.to_vec();
    let tensor_count = backend.context_tensors(1).len();
    assert!(backend.build_backward(1, invalid, None).is_err());
    assert_eq!(backend.context_tensors(1).len(), tensor_count);
    assert_eq!(backend.graph_nodes(invalid)?.0, nodes);
    assert_eq!(backend.graph_gradient(invalid, x)?, None);
    assert_eq!(backend.graph_gradient(valid, x)?, existing_gradient);
    backend.reset_graph(valid)?;
    backend.compute(valid)?;
    close(&gradient(&backend, valid, x)?, &[2.0, 4.0], 0.0);
    Ok(())
}

#[test]
fn backward_rejects_missing_storage_nonleaf_parameters_and_capacity_exhaustion() -> Result<()> {
    let mut backend = CpuBackend::new();
    let x = input(&mut backend, [2, 1, 1, 1], &[1.0, 2.0], true)?;
    assert!(
        backend
            .call("ggml_set_loss", &[Argument::Tensor(x)])
            .is_err()
    );
    let square = unary(&mut backend, "ggml_sqr", x)?;
    assert!(
        backend
            .call("ggml_set_param", &[Argument::Tensor(square)])
            .is_err()
    );
    let loss = unary(&mut backend, "ggml_sum", square)?;
    backend.call("ggml_set_loss", &[Argument::Tensor(loss)])?;
    for (capacity, gradients_enabled) in [(3, true), (512, false)] {
        let Observation::Graph(graph) = backend.call(
            "ggml_new_graph_custom",
            &[
                Argument::Context(1),
                Argument::Integer(capacity),
                Argument::Bool(gradients_enabled),
            ],
        )?
        else {
            return Err("graph result missing".into());
        };
        backend.call(
            "ggml_build_forward_expand",
            &[Argument::Graph(graph), Argument::Tensor(loss)],
        )?;
        let count = backend.context_tensors(1).len();
        assert!(backend.build_backward(1, graph, None).is_err());
        assert_eq!(backend.context_tensors(1).len(), count);
        assert_eq!(backend.graph_nodes(graph)?.0, [x, square, loss]);
        assert_eq!(backend.graph_gradient(graph, x)?, None);
    }
    Ok(())
}
