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
