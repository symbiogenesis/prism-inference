//! Same-format Q8 sibling arithmetic and atomic rejection controls.

use std::sync::Arc;

use super::super::{MatvecBackend, MatvecScratch, PQ2, PTQ1, RowParallelism, matvec_into};
use crate::{Result, model::Tensor};

fn weight(kind: u32, columns: usize, rows: usize) -> Result<Tensor> {
    let mut data = Vec::new();
    for index in 0..columns / 128 * rows {
        let mut block = [if index % 2 == 0 { 0x66 } else { 0x99 }; 34];
        block[..2].copy_from_slice(&[0x3555_u16, 0xb800, 0, 0x0001][index % 4].to_le_bytes());
        if kind == PTQ1 {
            data.extend_from_slice(&bonsai_ternary::pack_ptq(&block)?);
        } else {
            // Include generic PQ2 +2 codes, which cannot be encoded as PTQ1.
            block[2] = 0xe4;
            data.extend_from_slice(&block);
        }
    }
    Ok(Tensor {
        name: "q8-sibling-control".to_owned(),
        dimensions: vec![columns, rows],
        kind,
        data: Arc::from(data),
    })
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

#[test]
fn q8_siblings_match_independent_rows_for_pinned_source_and_fallback() -> Result<()> {
    for (kind, columns) in [(PQ2, 128), (PQ2, 384), (PTQ1, 256), (PTQ1, 384)] {
        let tensors = [
            weight(kind, columns, 33)?,
            weight(kind, columns, 9)?,
            weight(kind, columns, 5)?,
        ];
        let input: Vec<_> = (0..columns)
            .map(|index| [127.0, -128.0, 0.5001, -0.0, 63.5001, 1.5, -0.5001][index % 7])
            .collect();
        for pinned in [false, true] {
            for backend in [
                MatvecBackend::Scalar,
                MatvecBackend::Auto,
                MatvecBackend::ExperimentalNeon,
            ] {
                let make_scratch = || {
                    if pinned {
                        MatvecScratch::with_pinned_backend(backend)
                    } else {
                        MatvecScratch::with_backend(backend)
                    }
                };
                let mut expected: [Vec<f32>; 3] = std::array::from_fn(|_| Vec::new());
                for (tensor, output) in tensors.iter().zip(&mut expected) {
                    let mut reference = if pinned {
                        MatvecScratch::with_pinned_backend(MatvecBackend::Scalar)
                    } else {
                        MatvecScratch::with_backend(MatvecBackend::Scalar)
                    };
                    matvec_into(tensor, &input, output, &mut reference)?;
                }
                let workers = if cfg!(feature = "parallel-rows") {
                    4
                } else {
                    1
                };
                let mut scratch = make_scratch();
                scratch.set_row_parallelism(RowParallelism::new(workers)?);
                let mut outputs: [Vec<f32>; 3] = std::array::from_fn(|_| vec![f32::NAN]);
                let mut prepared = scratch.prepare(&tensors[0], &input)?;
                prepared.matvec_q8_siblings_into(&tensors.each_ref(), &mut outputs.each_mut())?;
                for (actual, expected) in outputs.iter().zip(&expected) {
                    assert_eq!(bits(actual), bits(expected));
                }
                assert_eq!(scratch.preparation_count(), 1);
                assert_eq!(scratch.statistics().matrix_successes, 3);
                assert_eq!(
                    scratch.statistics().grouped_row_dispatches,
                    u64::from(workers > 1)
                );
            }
        }
    }
    Ok(())
}

#[test]
fn q8_siblings_reject_late_geometry_scales_and_codes_then_recover() -> Result<()> {
    for (kind, columns) in [(PQ2, 384), (PTQ1, 256)] {
        let tensors = [
            weight(kind, columns, 33)?,
            weight(kind, columns, 9)?,
            weight(kind, columns, 5)?,
        ];
        let input = vec![1.5; columns];
        let mut scratch = MatvecScratch::with_pinned_backend(MatvecBackend::Auto);
        let workers = if cfg!(feature = "parallel-rows") {
            4
        } else {
            1
        };
        scratch.set_row_parallelism(RowParallelism::new(workers)?);
        let mut outputs: [Vec<f32>; 3] = std::array::from_fn(|_| vec![f32::NAN]);
        let mut malformed = tensors[2].clone();
        malformed.data = Arc::from(&malformed.data[..malformed.data.len() - 1]);
        let mut prepared = scratch.prepare(&tensors[0], &input)?;
        assert!(
            prepared
                .matvec_q8_siblings_into(
                    &[&tensors[0], &tensors[1], &malformed],
                    &mut outputs.each_mut()
                )
                .is_err()
        );
        assert!(outputs.iter().all(Vec::is_empty));
        assert!(!prepared.scratch.diagnostics().dispatched);
        for corrupt_code in [false, true] {
            let mut failing = tensors[2].clone();
            let mut data = failing.data.to_vec();
            let len = data.len();
            if kind == PTQ1 && corrupt_code {
                // A zero-scale block still has to reject noncanonical trits.
                data[len - 28] = 255;
                data[len - 2..].copy_from_slice(&0_u16.to_le_bytes());
            } else {
                let scale = if kind == PTQ1 { len - 2 } else { len - 34 };
                data[scale..scale + 2].copy_from_slice(&0x7c00_u16.to_le_bytes());
            }
            failing.data = Arc::from(data);
            assert!(
                prepared
                    .matvec_q8_siblings_into(
                        &[&tensors[0], &tensors[1], &failing],
                        &mut outputs.each_mut()
                    )
                    .is_err()
            );
            assert!(outputs.iter().all(Vec::is_empty));
        }
        prepared.matvec_q8_siblings_into(&tensors.each_ref(), &mut outputs.each_mut())?;
        assert!(outputs.iter().all(|output| !output.is_empty()));
        assert_eq!(scratch.preparation_count(), 1);
        assert_eq!(scratch.statistics().matrix_successes, 3);
    }
    Ok(())
}

#[test]
fn q8_siblings_refuse_mixed_formats_and_q8k_preparation_before_dispatch() -> Result<()> {
    let odd = weight(PQ2, 384, 9)?;
    let ptq1 = weight(PTQ1, 384, 5)?;
    let even = weight(PQ2, 256, 9)?;
    for pinned in [false, true] {
        let mut scratch = if pinned {
            MatvecScratch::with_pinned_backend(MatvecBackend::Auto)
        } else {
            MatvecScratch::with_backend(MatvecBackend::Auto)
        };
        for (first, second) in [(&odd, &ptq1), (&even, &even)] {
            let input = vec![1.0; first.dimensions[0]];
            let mut outputs: [Vec<f32>; 2] = std::array::from_fn(|_| vec![f32::NAN]);
            let mut prepared = scratch.prepare(first, &input)?;
            assert!(
                prepared
                    .matvec_q8_siblings_into(&[first, second], &mut outputs.each_mut())
                    .is_err()
            );
            assert!(outputs.iter().all(Vec::is_empty));
            assert!(!scratch.diagnostics().dispatched);
        }
        assert_eq!(scratch.statistics().matrix_successes, 0);
        assert_eq!(scratch.statistics().grouped_row_dispatches, 0);
    }
    Ok(())
}
