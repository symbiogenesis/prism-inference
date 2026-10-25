//! Focused image-gradient overlap, padding, dilation, batch and stride regressions.

use bonsai_prism_runtime::Result;
use bonsai_prism_runtime::backend::{Argument, CpuBackend, Observation, TensorId};

fn tensor_result(value: &Observation) -> Result<TensorId> {
    match value {
        Observation::Tensor(Some(tensor)) => Ok(*tensor),
        _ => Err(std::io::Error::other("expected tensor result").into()),
    }
}

fn evaluate(backend: &mut CpuBackend, tensor: TensorId) -> Result<Vec<f32>> {
    let Observation::Graph(graph) = backend.call("ggml_new_graph", &[Argument::Context(1)])? else {
        return Err(std::io::Error::other("expected graph result").into());
    };
    backend.call(
        "ggml_build_forward_expand",
        &[Argument::Graph(graph), Argument::Tensor(tensor)],
    )?;
    assert!(backend.supports(tensor)?);
    backend.compute(graph)?;
    backend.values(tensor)
}

fn im2col_backward(
    backend: &mut CpuBackend,
    gradient: TensorId,
    kernel: TensorId,
    shape: [i64; 4],
    parameters: [i64; 6],
    two_dimensional: bool,
) -> Result<TensorId> {
    let mut arguments = vec![
        Argument::Context(1),
        Argument::Tensor(gradient),
        Argument::Tensor(kernel),
        Argument::Integers(shape.to_vec()),
    ];
    arguments.extend(parameters.map(Argument::Integer));
    arguments.push(Argument::Bool(two_dimensional));
    tensor_result(&backend.call("ggml_im2col_back", &arguments)?)
}

fn pool_backward(
    backend: &mut CpuBackend,
    gradient: TensorId,
    forward: TensorId,
    average: bool,
    kernel: [i64; 2],
    padding: f32,
) -> Result<TensorId> {
    tensor_result(&backend.call(
        "ggml_pool_2d_back",
        &[
            Argument::Context(1),
            Argument::Tensor(gradient),
            Argument::Tensor(forward),
            Argument::Integer(i64::from(average)),
            Argument::Integer(kernel[0]),
            Argument::Integer(kernel[1]),
            Argument::Integer(1),
            Argument::Integer(1),
            Argument::Float(padding),
            Argument::Float(padding),
        ],
    )?)
}

#[test]
fn im2col_backward_accumulates_channels_and_batches_in_one_dimension() -> Result<()> {
    let mut backend = CpuBackend::new();
    let gradient = backend.tensor(1, 0, [4, 3, 2, 1])?;
    let data = (1_u8..=24).map(f32::from).collect::<Vec<_>>();
    backend.set_values(gradient, &data)?;
    let kernel = backend.tensor(1, 1, [2, 2, 1, 1])?;
    let backward = im2col_backward(
        &mut backend,
        gradient,
        kernel,
        [4, 2, 2, 1],
        [1, 0, 0, 0, 1, 0],
        false,
    )?;
    assert_eq!(backend.info(backward)?.kind, 0);
    assert_eq!(
        evaluate(&mut backend, backward)?,
        [
            1.0, 7.0, 15.0, 10.0, 3.0, 11.0, 19.0, 12.0, 13.0, 31.0, 39.0, 22.0, 15.0, 35.0, 43.0,
            24.0
        ]
    );
    assert_eq!(backend.values(gradient)?, data);
    Ok(())
}

#[test]
fn im2col_backward_respects_stride_dilation_and_padded_positions() -> Result<()> {
    let mut backend = CpuBackend::new();
    let gradient = backend.tensor(1, 0, [4, 3, 3, 1])?;
    backend.set_values(gradient, &(1_u8..=36).map(f32::from).collect::<Vec<_>>())?;
    let kernel = backend.tensor(1, 0, [2, 2, 1, 1])?;
    let backward = im2col_backward(
        &mut backend,
        gradient,
        kernel,
        [5, 5, 1, 1],
        [2, 2, 1, 1, 2, 2],
        true,
    )?;
    assert_eq!(
        evaluate(&mut backend, backward)?,
        [
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 42.0, 0.0, 58.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 90.0,
            0.0, 106.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ]
    );
    Ok(())
}

#[test]
fn im2col_backward_preserves_native_kernel_accumulation_order() -> Result<()> {
    let mut backend = CpuBackend::new();
    let gradient = backend.tensor(1, 0, [3, 3, 1, 1])?;
    backend.set_values(
        gradient,
        &[0.0, 0.0, 3.0, 0.0, -1.0e20, 0.0, 1.0e20, 0.0, 0.0],
    )?;
    let kernel = backend.tensor(1, 0, [3, 1, 1, 1])?;
    let backward = im2col_backward(
        &mut backend,
        gradient,
        kernel,
        [5, 1, 1, 1],
        [1, 0, 0, 0, 1, 0],
        false,
    )?;
    assert_eq!(evaluate(&mut backend, backward)?, [0.0, 0.0, 3.0, 0.0, 0.0]);
    Ok(())
}

#[test]
fn im2col_backward_reads_logical_strided_gradients() -> Result<()> {
    let mut backend = CpuBackend::new();
    let storage = backend.tensor(1, 0, [3, 4, 1, 1])?;
    backend.set_values(
        storage,
        &[
            1.0, 5.0, 9.0, 2.0, 6.0, 10.0, 3.0, 7.0, 11.0, 4.0, 8.0, 12.0,
        ],
    )?;
    let gradient = tensor_result(&backend.call(
        "ggml_transpose",
        &[Argument::Context(1), Argument::Tensor(storage)],
    )?)?;
    let kernel = backend.tensor(1, 0, [2, 2, 1, 1])?;
    let backward = im2col_backward(
        &mut backend,
        gradient,
        kernel,
        [4, 2, 1, 1],
        [1, 0, 0, 0, 1, 0],
        false,
    )?;
    assert_eq!(
        evaluate(&mut backend, backward)?,
        [1.0, 7.0, 15.0, 10.0, 3.0, 11.0, 19.0, 12.0]
    );
    Ok(())
}

#[test]
fn max_pool_backward_selects_first_tied_maximum_and_accumulates_overlaps() -> Result<()> {
    let mut backend = CpuBackend::new();
    let forward = backend.tensor(1, 0, [3, 3, 1, 1])?;
    let gradient = backend.tensor(1, 0, [2, 2, 1, 1])?;
    backend.set_values(gradient, &[1.0, 2.0, 3.0, 4.0])?;
    backend.set_values(forward, &[9.0, 9.0, 1.0, 9.0, 9.0, 1.0, 1.0, 1.0, 1.0])?;
    let backward = pool_backward(&mut backend, gradient, forward, false, [2, 2], 0.0)?;
    assert_eq!(
        evaluate(&mut backend, backward)?,
        [1.0, 2.0, 0.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0]
    );
    backend.set_values(forward, &[1.0, 2.0, 3.0, 4.0, 9.0, 5.0, 6.0, 7.0, 8.0])?;
    assert_eq!(
        evaluate(&mut backend, backward)?,
        [0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0]
    );
    Ok(())
}

#[test]
fn average_pool_backward_includes_padding_in_divisor_and_separates_planes() -> Result<()> {
    let mut backend = CpuBackend::new();
    let forward = backend.tensor(1, 0, [2, 2, 2, 2])?;
    let gradient = backend.tensor(1, 0, [2, 2, 2, 2])?;
    backend.set_values(
        gradient,
        &[
            9.0, 18.0, 27.0, 36.0, 18.0, 36.0, 54.0, 72.0, 27.0, 54.0, 81.0, 108.0, 36.0, 72.0,
            108.0, 144.0,
        ],
    )?;
    let backward = pool_backward(&mut backend, gradient, forward, true, [3, 3], 1.75)?;
    assert_eq!(
        evaluate(&mut backend, backward)?,
        [
            10.0, 10.0, 10.0, 10.0, 20.0, 20.0, 20.0, 20.0, 30.0, 30.0, 30.0, 30.0, 40.0, 40.0,
            40.0, 40.0
        ]
    );
    Ok(())
}

#[test]
fn max_pool_backward_retains_native_nonfinite_comparison_rules() -> Result<()> {
    let mut backend = CpuBackend::new();
    let forward = backend.tensor(1, 0, [2, 2, 1, 1])?;
    let gradient = backend.tensor(1, 0, [1; 4])?;
    backend.set_values(gradient, &[5.0])?;
    backend.set_values(forward, &[f32::NAN, 1.0, f32::NAN, 2.0])?;
    let backward = pool_backward(&mut backend, gradient, forward, false, [2, 2], 0.0)?;
    assert_eq!(evaluate(&mut backend, backward)?, [0.0, 0.0, 0.0, 5.0]);
    backend.set_values(forward, &[f32::NEG_INFINITY; 4])?;
    assert_eq!(evaluate(&mut backend, backward)?, [0.0; 4]);
    Ok(())
}

#[test]
fn backward_image_rejects_malformed_geometry_and_nonscalar_storage() -> Result<()> {
    let mut backend = CpuBackend::new();
    let gradient = backend.tensor(1, 0, [1; 4])?;
    let kernel = backend.tensor(1, 0, [1; 4])?;
    let backward = im2col_backward(
        &mut backend,
        gradient,
        kernel,
        [1; 4],
        [1, 0, 0, 0, 1, 0],
        false,
    )?;
    backend.info_mut(backward)?.parameters[..4].copy_from_slice(&0_i32.to_le_bytes());
    assert!(evaluate(&mut backend, backward).is_err());
    let backward = pool_backward(&mut backend, gradient, kernel, false, [1, 1], 0.0)?;
    backend.info_mut(backward)?.parameters[..4].copy_from_slice(&2_i32.to_le_bytes());
    assert!(evaluate(&mut backend, backward).is_err());
    let half_gradient = backend.tensor(1, 1, [1; 4])?;
    let rejected = im2col_backward(
        &mut backend,
        half_gradient,
        kernel,
        [1; 4],
        [1, 0, 0, 0, 1, 0],
        false,
    );
    assert!(rejected.is_err());
    let integer_kernel = backend.tensor(1, 26, [1; 4])?;
    assert!(
        im2col_backward(
            &mut backend,
            gradient,
            integer_kernel,
            [1; 4],
            [1, 0, 0, 0, 1, 0],
            false,
        )
        .is_err()
    );
    assert!(pool_backward(&mut backend, half_gradient, kernel, false, [1, 1], 0.0).is_err());
    assert!(pool_backward(&mut backend, gradient, half_gradient, false, [1, 1], 0.0).is_err());
    Ok(())
}
