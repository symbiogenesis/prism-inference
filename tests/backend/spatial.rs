//! Focused regressions for pinned window, relative position and scalar-add contracts.

use bonsai_prism_runtime::Result;
use bonsai_prism_runtime::backend::{Argument, CpuBackend, Observation, TensorId};

fn operation(backend: &mut CpuBackend, name: &str, args: &[Argument]) -> Result<TensorId> {
    match backend.call(name, args)? {
        Observation::Tensor(Some(tensor)) => Ok(tensor),
        _ => Err(std::io::Error::other("expected tensor").into()),
    }
}

fn compute(backend: &mut CpuBackend, tensor: TensorId) -> Result<()> {
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
fn spatial_windows_pad_partial_patches_and_crop_on_reconstruction() -> Result<()> {
    let mut backend = CpuBackend::new();
    let image = backend.tensor(1, 0, [1, 3, 2, 1])?;
    backend.set_values(image, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;
    let patches = operation(
        &mut backend,
        "ggml_win_part",
        &[
            Argument::Context(1),
            Argument::Tensor(image),
            Argument::Integer(2),
        ],
    )?;
    assert_eq!(backend.info(patches)?.shape, [1, 2, 2, 2]);
    assert_eq!(
        i32::from_le_bytes(backend.info(patches)?.parameters[0..4].try_into()?),
        2
    );
    compute(&mut backend, patches)?;
    assert_eq!(
        backend.values(patches)?,
        [1.0, 2.0, 4.0, 5.0, 3.0, 0.0, 6.0, 0.0]
    );
    let restored = operation(
        &mut backend,
        "ggml_win_unpart",
        &[
            Argument::Context(1),
            Argument::Tensor(patches),
            Argument::Integer(3),
            Argument::Integer(2),
            Argument::Integer(2),
        ],
    )?;
    compute(&mut backend, restored)?;
    assert_eq!(backend.values(restored)?, backend.values(image)?);
    Ok(())
}

#[test]
fn spatial_windows_keep_channels_and_both_patch_axes_in_native_order() -> Result<()> {
    let mut backend = CpuBackend::new();
    let image = backend.tensor(1, 0, [2, 1, 3, 1])?;
    backend.set_values(image, &[1.0, 11.0, 2.0, 12.0, 3.0, 13.0])?;
    let patches = operation(
        &mut backend,
        "ggml_win_part",
        &[
            Argument::Context(1),
            Argument::Tensor(image),
            Argument::Integer(2),
        ],
    )?;
    compute(&mut backend, patches)?;
    assert_eq!(
        backend.values(patches)?,
        [
            1.0, 11.0, 0.0, 0.0, 2.0, 12.0, 0.0, 0.0, 3.0, 13.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ]
    );
    Ok(())
}

#[test]
fn spatial_relative_positions_copy_f16_and_bf16_payloads_without_conversion() -> Result<()> {
    for kind in [1, 30] {
        let mut backend = CpuBackend::new();
        let positions = backend.tensor(1, kind, [2, 3, 1, 1])?;
        let payload = [0x8000_u16, 0x7c01, 0x3c00, 0x4000, 0x4200, 0x4400];
        let bytes: Vec<_> = payload
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        backend.set_values(positions, &[0.0; 6])?;
        backend.set(positions, 0, &bytes)?;
        let expanded = operation(
            &mut backend,
            "ggml_get_rel_pos",
            &[
                Argument::Context(1),
                Argument::Tensor(positions),
                Argument::Integer(2),
                Argument::Integer(2),
            ],
        )?;
        assert_eq!(backend.info(expanded)?.kind, 1);
        assert_eq!(backend.info(expanded)?.shape, [2, 2, 2, 1]);
        compute(&mut backend, expanded)?;
        let expected: Vec<_> = [
            0x3c00_u16, 0x4000, 0x8000, 0x7c01, 0x4200, 0x4400, 0x3c00, 0x4000,
        ]
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
        assert_eq!(backend.get(expanded, 0, expected.len())?, expected);
    }
    Ok(())
}

#[test]
fn spatial_relative_addition_preserves_heads_queries_and_inplace_aliases() -> Result<()> {
    for inplace in [false, true] {
        let mut backend = CpuBackend::new();
        let attention = backend.tensor(1, 0, [4, 2, 2, 1])?;
        let values: Vec<_> = (1_u8..=16).map(f32::from).collect();
        backend.set_values(attention, &values)?;
        let width = backend.tensor(1, 0, [2, 1, 2, 2])?;
        let height = backend.tensor(1, 0, [2, 1, 2, 2])?;
        backend.set_values(width, &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0])?;
        backend.set_values(
            height,
            &[100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0],
        )?;
        let result = operation(
            &mut backend,
            if inplace {
                "ggml_add_rel_pos_inplace"
            } else {
                "ggml_add_rel_pos"
            },
            &[
                Argument::Context(1),
                Argument::Tensor(attention),
                Argument::Tensor(width),
                Argument::Tensor(height),
            ],
        )?;
        compute(&mut backend, result)?;
        let expected = [
            111.0, 122.0, 213.0, 224.0, 335.0, 346.0, 437.0, 448.0, 559.0, 570.0, 661.0, 672.0,
            783.0, 794.0, 885.0, 896.0,
        ];
        assert_eq!(backend.values(result)?, expected);
        if inplace {
            assert_eq!(backend.values(attention)?, expected);
        } else {
            assert_eq!(backend.values(attention)?, values);
        }
    }
    Ok(())
}

#[test]
fn spatial_relative_addition_keeps_native_f32_addition_order() -> Result<()> {
    let mut backend = CpuBackend::new();
    let attention = backend.tensor(1, 0, [1; 4])?;
    let width = backend.tensor(1, 0, [1; 4])?;
    let height = backend.tensor(1, 0, [1; 4])?;
    backend.set_values(attention, &[1e20])?;
    backend.set_values(width, &[1.0])?;
    backend.set_values(height, &[-1e20])?;
    let result = operation(
        &mut backend,
        "ggml_add_rel_pos",
        &[
            Argument::Context(1),
            Argument::Tensor(attention),
            Argument::Tensor(width),
            Argument::Tensor(height),
        ],
    )?;
    compute(&mut backend, result)?;
    assert_eq!(backend.values(result)?, [1.0]);
    Ok(())
}

#[test]
fn spatial_relative_addition_preserves_native_untouched_extra_slabs() -> Result<()> {
    let mut backend = CpuBackend::new();
    let attention = backend.tensor(1, 0, [1, 1, 1, 2])?;
    let width = backend.tensor(1, 0, [1; 4])?;
    let height = backend.tensor(1, 0, [1; 4])?;
    backend.set_values(attention, &[1.0, 99.0])?;
    backend.set_values(width, &[10.0])?;
    backend.set_values(height, &[100.0])?;
    let result = operation(
        &mut backend,
        "ggml_add_rel_pos",
        &[
            Argument::Context(1),
            Argument::Tensor(attention),
            Argument::Tensor(width),
            Argument::Tensor(height),
        ],
    )?;
    compute(&mut backend, result)?;
    assert_eq!(backend.values(result)?, [111.0, 99.0]);
    Ok(())
}

#[test]
fn spatial_scalar_add_supports_floating_types_and_padded_inplace_rows() -> Result<()> {
    for (kind, scalar_kind) in [(0, 0), (1, 0), (1, 1), (30, 0), (30, 30)] {
        let mut backend = CpuBackend::new();
        let data = backend.tensor(1, kind, [3, 2, 1, 1])?;
        let scalar = backend.tensor(1, scalar_kind, [1; 4])?;
        backend.set_values(data, &[1.0, 2.0, 99.0, 3.0, 4.0, 99.0])?;
        backend.set_values(scalar, &[0.5])?;
        let width = i64::try_from(backend.info(data)?.strides[1])?;
        let view = operation(
            &mut backend,
            "ggml_view_2d",
            &[
                Argument::Context(1),
                Argument::Tensor(data),
                Argument::Integer(2),
                Argument::Integer(2),
                Argument::Integer(width),
                Argument::Integer(0),
            ],
        )?;
        let sum = operation(
            &mut backend,
            "ggml_add1_inplace",
            &[
                Argument::Context(1),
                Argument::Tensor(view),
                Argument::Tensor(scalar),
            ],
        )?;
        compute(&mut backend, sum)?;
        assert_eq!(backend.values(data)?, [1.5, 2.5, 99.0, 3.5, 4.5, 99.0]);
    }
    Ok(())
}

#[test]
fn spatial_constructors_reject_invalid_geometry_and_scalar_domains() -> Result<()> {
    let mut backend = CpuBackend::new();
    let image = backend.tensor(1, 0, [1, 3, 2, 1])?;
    for window in [0, -1] {
        assert!(
            backend
                .call(
                    "ggml_win_part",
                    &[
                        Argument::Context(1),
                        Argument::Tensor(image),
                        Argument::Integer(window),
                    ]
                )
                .is_err()
        );
    }
    let patches = backend.tensor(1, 0, [1, 2, 2, 1])?;
    assert!(
        backend
            .call(
                "ggml_win_unpart",
                &[
                    Argument::Context(1),
                    Argument::Tensor(patches),
                    Argument::Integer(3),
                    Argument::Integer(2),
                    Argument::Integer(2),
                ]
            )
            .is_err()
    );
    let positions = backend.tensor(1, 1, [1, 3, 1, 1])?;
    assert!(
        backend
            .call(
                "ggml_get_rel_pos",
                &[
                    Argument::Context(1),
                    Argument::Tensor(positions),
                    Argument::Integer(2),
                    Argument::Integer(1),
                ]
            )
            .is_err()
    );
    assert!(
        backend
            .call(
                "ggml_add1",
                &[
                    Argument::Context(1),
                    Argument::Tensor(image),
                    Argument::Tensor(patches),
                ]
            )
            .is_err()
    );
    Ok(())
}
