use std::sync::Arc;

use super::*;

fn tensor(columns: usize, rows: usize, symbol: u8) -> Tensor {
    let mut block = [symbol; 34];
    block[..2].copy_from_slice(&0x3555_u16.to_le_bytes());
    Tensor {
        name: format!("sibling-{rows}"),
        dimensions: vec![columns, rows],
        kind: PQ2,
        data: Arc::from(block.repeat(columns / 128 * rows)),
    }
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

fn bf16_tensor(columns: usize, rows: usize) -> Tensor {
    let data: Vec<u8> = [0x3f80_u16, 0xc000, 0x8000, 0x0080, 0x3eaa, 0x4000, 0x0001]
        .into_iter()
        .cycle()
        .take(columns * rows)
        .flat_map(u16::to_le_bytes)
        .collect();
    Tensor {
        name: format!("bf16-sibling-{rows}"),
        dimensions: vec![columns, rows],
        kind: BF16,
        data: Arc::from(data),
    }
}

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "Keep the independent scalar oracle, scoped reuse, late failure and recovery controls together."
)]
fn tiled_bf16_siblings_match_scalar_reuse_and_clear_late_failure() -> Result<()> {
    let workers = if cfg!(feature = "parallel-rows") {
        4
    } else {
        1
    };
    for columns in [20, 256] {
        let tensors = [
            bf16_tensor(columns, 17),
            bf16_tensor(columns, 9),
            bf16_tensor(columns, 3),
        ];
        let inputs: Vec<Vec<f32>> = (0..9)
            .map(|token| {
                (0..columns)
                    .map(|index| [1.5, -128.0, 0.0, -0.0, 63.5, -3.25][(index + token) % 6])
                    .collect()
            })
            .collect();
        for backend in [MatvecBackend::Scalar, MatvecBackend::Auto] {
            for pinned in [false, true] {
                let mut tile = ProjectionTile::with_backend(backend);
                tile.set_row_parallelism(RowParallelism::new(workers)?);
                let mut outputs: [Vec<Vec<f32>>; 3] =
                    std::array::from_fn(|_| vec![Vec::new(); inputs.len()]);
                for previous in [None, Some(&tensors[2])] {
                    let [a, b, c] = &mut outputs;
                    tile.project_siblings_with_previous(
                        previous,
                        &tensors.each_ref(),
                        &inputs,
                        &mut [a, b, c],
                        pinned,
                        &|| Ok(()),
                    )?;
                    for (tensor, group) in tensors.iter().zip(&outputs) {
                        for (input, output) in inputs.iter().zip(group) {
                            let mut scratch = MatvecScratch::with_backend(MatvecBackend::Scalar);
                            let mut expected = Vec::new();
                            matvec_into(tensor, input, &mut expected, &mut scratch)?;
                            assert_eq!(bits(output), bits(&expected));
                        }
                    }
                }
                assert!(
                    tile.preparations
                        .iter()
                        .all(|scratch| scratch.preparation_count() == 1)
                );
                assert_eq!(tile.packed_projections, 0);
                assert_eq!(tile.decoded_blocks, 0);
                let expected: Vec<Vec<Vec<u32>>> = outputs
                    .iter()
                    .map(|group| group.iter().map(|row| bits(row)).collect())
                    .collect();
                let mut failing = bf16_tensor(columns, 3);
                let mut data = failing.data.to_vec();
                let end = data.len();
                data[end - 2..].copy_from_slice(&0x7f80_u16.to_le_bytes());
                failing.data = Arc::from(data);
                let [a, b, c] = &mut outputs;
                assert!(
                    tile.project_siblings_with_previous(
                        None,
                        &[&tensors[0], &tensors[1], &failing],
                        &inputs,
                        &mut [a, b, c],
                        pinned,
                        &|| Ok(()),
                    )
                    .is_err()
                );
                assert!(outputs.iter().flatten().all(Vec::is_empty));
                let [a, b, c] = &mut outputs;
                tile.project_siblings_with_previous(
                    None,
                    &tensors.each_ref(),
                    &inputs,
                    &mut [a, b, c],
                    pinned,
                    &|| Ok(()),
                )?;
                let recovered: Vec<Vec<Vec<u32>>> = outputs
                    .iter()
                    .map(|group| group.iter().map(|row| bits(row)).collect())
                    .collect();
                assert_eq!(recovered, expected);
                let [a, b, c] = &mut outputs;
                assert!(
                    tile.project_siblings_with_previous(
                        None,
                        &tensors.each_ref(),
                        &inputs,
                        &mut [a, b, c],
                        pinned,
                        &|| Err(invalid("cancel BF16 siblings")),
                    )
                    .is_err()
                );
                assert!(outputs.iter().flatten().all(Vec::is_empty));
            }
        }
        let wrong_width = bf16_tensor(columns + 1, 5);
        let packed = tensor(256, 5, 0xe4);
        assert!(!ProjectionTile::supports_siblings(&[
            &tensors[0],
            &wrong_width
        ]));
        assert!(!ProjectionTile::supports_siblings(&[&tensors[0], &packed]));
    }
    Ok(())
}

#[test]
fn x1_tiled_cpu_siblings_match_independent_matvec_and_clear_late_failure() -> Result<()> {
    let workers = if cfg!(feature = "parallel-rows") {
        4
    } else {
        1
    };
    for columns in [256, 512] {
        let tensors = [
            tensor(columns, 13, 0xe4),
            tensor(columns, 9, 0x1b),
            tensor(columns, 5, 0x72),
        ];
        let inputs: Vec<Vec<f32>> = (0..9)
            .map(|token| {
                (0..columns)
                    .map(|index| [1.5, -128.0, 0.0, -0.0, 63.5, -3.25][(index + token) % 6])
                    .collect()
            })
            .collect();
        for backend in [MatvecBackend::Scalar, MatvecBackend::Auto] {
            for pinned in [false, true] {
                let mut tile = ProjectionTile::with_backend(backend);
                tile.set_row_parallelism(RowParallelism::new(workers)?);
                let mut outputs: [Vec<Vec<f32>>; 3] =
                    std::array::from_fn(|_| vec![Vec::new(); inputs.len()]);
                let [a, b, c] = &mut outputs;
                tile.project_siblings_with_previous(
                    None,
                    &tensors.each_ref(),
                    &inputs,
                    &mut [a, b, c],
                    pinned,
                    &|| Ok(()),
                )?;
                assert_eq!(tile.packed_projections, 3);
                assert_eq!(tile.decoded_blocks, u64::try_from(columns / 128 * 27)?);
                for (tensor, group) in tensors.iter().zip(&outputs) {
                    for (input, output) in inputs.iter().zip(group) {
                        let mut scratch = if pinned {
                            MatvecScratch::with_pinned_backend(MatvecBackend::Scalar)
                        } else {
                            MatvecScratch::with_backend(MatvecBackend::Scalar)
                        };
                        let mut expected = Vec::new();
                        matvec_into(tensor, input, &mut expected, &mut scratch)?;
                        assert_eq!(bits(output), bits(&expected));
                    }
                }
                let mut failing = tensor(columns, 5, 0x72);
                let mut data = failing.data.to_vec();
                let last_scale = data.len() - 34;
                data[last_scale..last_scale + 2].copy_from_slice(&0x7c00_u16.to_le_bytes());
                failing.data = Arc::from(data);
                let [a, b, c] = &mut outputs;
                assert!(
                    tile.project_siblings_with_previous(
                        None,
                        &[&tensors[0], &tensors[1], &failing],
                        &inputs,
                        &mut [a, b, c],
                        pinned,
                        &|| Ok(()),
                    )
                    .is_err()
                );
                assert!(outputs.iter().flatten().all(Vec::is_empty));
                let [a, b, c] = &mut outputs;
                assert!(
                    tile.project_siblings_with_previous(
                        None,
                        &tensors.each_ref(),
                        &inputs,
                        &mut [a, b, c],
                        pinned,
                        &|| Err(invalid("cancel siblings")),
                    )
                    .is_err()
                );
                assert!(outputs.iter().flatten().all(Vec::is_empty));
            }
        }
        let odd = tensor(384, 5, 0x72);
        assert!(!ProjectionTile::supports_siblings(&[&tensors[0], &odd]));
        let mut tile = ProjectionTile::default();
        let mut a = vec![vec![1.0]; inputs.len()];
        let mut b = a.clone();
        assert!(
            tile.project_siblings_with_previous(
                None,
                &[&tensors[0], &odd],
                &inputs,
                &mut [&mut a, &mut b],
                true,
                &|| Ok(()),
            )
            .is_err()
        );
        assert!(a.iter().chain(&b).all(Vec::is_empty));
    }
    Ok(())
}
