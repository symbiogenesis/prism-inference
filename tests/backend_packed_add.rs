//! Focused packed scalar-add storage, encoding and alias regressions.

use bonsai_prism_runtime::Result;
use bonsai_prism_runtime::backend::{Argument, CpuBackend, Observation, TensorId};

fn operation(backend: &mut CpuBackend, name: &str, args: &[Argument]) -> Result<TensorId> {
    match backend.call(name, args)? {
        Observation::Tensor(Some(tensor)) => Ok(tensor),
        _ => Err(std::io::Error::other("expected tensor").into()),
    }
}

fn evaluate(backend: &mut CpuBackend, tensor: TensorId) -> Result<()> {
    assert!(backend.supports(tensor)?);
    let Observation::Graph(graph) = backend.call("ggml_new_graph", &[Argument::Context(1)])? else {
        return Err(std::io::Error::other("expected graph").into());
    };
    backend.call(
        "ggml_build_forward_expand",
        &[Argument::Graph(graph), Argument::Tensor(tensor)],
    )?;
    backend.compute(graph)
}

#[test]
fn packed_add1_requantizes_every_row_and_preserves_alias_ownership() -> Result<()> {
    for kind in [
        2, 3, 6, 7, 8, 10, 11, 12, 13, 14, 20, 23, 34, 35, 39, 40, 41, 42, 142, 143,
    ] {
        for inplace in [false, true] {
            let mut backend = CpuBackend::new();
            let data = backend.tensor(1, kind, [256, 2, 2, 1])?;
            let scalar = backend.tensor(1, 0, [1; 4])?;
            let values: Vec<_> = [1.0, -1.0, 2.0, -2.0]
                .into_iter()
                .flat_map(|value| [value; 256])
                .collect();
            backend.set_values(data, &values)?;
            let before = backend.values(data)?;
            backend.set_values(scalar, &[0.5])?;
            let result = operation(
                &mut backend,
                if inplace {
                    "ggml_add1_inplace"
                } else {
                    "ggml_add1"
                },
                &[
                    Argument::Context(1),
                    Argument::Tensor(data),
                    Argument::Tensor(scalar),
                ],
            )?;
            evaluate(&mut backend, result)?;
            for (&value, expected) in backend.values(result)?.iter().zip(
                [1.5, -0.5, if kind == 39 { 2.0 } else { 2.5 }, -1.5]
                    .into_iter()
                    .flat_map(|value| [value; 256]),
            ) {
                assert!(
                    (value - expected).abs() < 0.15,
                    "packed type {kind}: {value} != {expected}"
                );
            }
            if inplace {
                assert_eq!(backend.values(data)?, backend.values(result)?);
            } else {
                assert_eq!(backend.values(data)?, before);
            }
        }
    }
    Ok(())
}

#[test]
fn packed_add1_preserves_padded_rows_and_rejects_unqualified_domains() -> Result<()> {
    let mut backend = CpuBackend::new();
    let data = backend.tensor(1, 41, [256, 2, 1, 1])?;
    let scalar = backend.tensor(1, 0, [1; 4])?;
    let values: Vec<_> = [1.0, 99.0, -1.0, 99.0]
        .into_iter()
        .flat_map(|value| [value; 128])
        .collect();
    backend.set_values(data, &values)?;
    backend.set_values(scalar, &[0.5])?;
    let view = operation(
        &mut backend,
        "ggml_view_2d",
        &[
            Argument::Context(1),
            Argument::Tensor(data),
            Argument::Integer(128),
            Argument::Integer(2),
            Argument::Integer(36),
            Argument::Integer(0),
        ],
    )?;
    assert_eq!(backend.info(view)?.strides, [18, 36, 72, 72]);
    let sum = operation(
        &mut backend,
        "ggml_add1_inplace",
        &[
            Argument::Context(1),
            Argument::Tensor(view),
            Argument::Tensor(scalar),
        ],
    )?;
    evaluate(&mut backend, sum)?;
    let expected: Vec<_> = [1.5, 99.0, -0.5, 99.0]
        .into_iter()
        .flat_map(|value| [value; 128])
        .collect();
    assert_eq!(backend.values(data)?, expected);
    let four_dimensional = backend.tensor(1, 41, [128, 1, 1, 2])?;
    assert!(
        backend
            .call(
                "ggml_add1",
                &[
                    Argument::Context(1),
                    Argument::Tensor(four_dimensional),
                    Argument::Tensor(scalar),
                ]
            )
            .is_err()
    );
    let half = backend.tensor(1, 1, [1; 4])?;
    assert!(
        backend
            .call(
                "ggml_add1",
                &[
                    Argument::Context(1),
                    Argument::Tensor(data),
                    Argument::Tensor(half),
                ]
            )
            .is_err()
    );
    for kind in [9, 15] {
        let auxiliary = backend.tensor(1, kind, [256, 1, 1, 1])?;
        assert!(
            backend
                .call(
                    "ggml_add1",
                    &[
                        Argument::Context(1),
                        Argument::Tensor(auxiliary),
                        Argument::Tensor(scalar),
                    ]
                )
                .is_err()
        );
    }
    Ok(())
}

#[test]
fn view_metadata_derives_unused_axes_from_actual_parent_strides() -> Result<()> {
    let mut backend = CpuBackend::new();
    let storage = backend.tensor(1, 0, [64, 1, 1, 1])?;
    let view = operation(
        &mut backend,
        "ggml_view_3d",
        &[
            Argument::Context(1),
            Argument::Tensor(storage),
            Argument::Integer(2),
            Argument::Integer(2),
            Argument::Integer(3),
            Argument::Integer(16),
            Argument::Integer(48),
            Argument::Integer(0),
        ],
    )?;
    assert_eq!(backend.info(view)?.strides, [4, 16, 48, 144]);
    assert_eq!(backend.info(view)?.bytes()?, 120);
    assert!(
        backend
            .call(
                "ggml_view_2d",
                &[
                    Argument::Context(1),
                    Argument::Tensor(storage),
                    Argument::Integer(1),
                    Argument::Integer(3),
                    Argument::Integer(i64::MAX),
                    Argument::Integer(0),
                ]
            )
            .is_err()
    );
    Ok(())
}
