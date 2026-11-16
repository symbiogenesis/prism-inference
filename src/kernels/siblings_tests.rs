//! Atomic failures and scheduling order for prepared CPU sibling execution.

use std::sync::Arc;

use super::super::{MatvecBackend, MatvecScratch, RowParallelism};
use crate::{Result, invalid, model::Tensor};

fn weight(rows: usize) -> Tensor {
    let mut block = [0x66; 34];
    block[..2].copy_from_slice(&0x3555_u16.to_le_bytes());
    Tensor {
        name: "sibling-control".to_owned(),
        dimensions: vec![256, rows],
        kind: 142,
        data: Arc::<[u8]>::from(block.repeat(2 * rows)).into(),
    }
}

#[test]
fn cpu_sibling_projection_rejects_late_geometry_and_recovers() -> Result<()> {
    let first = weight(17);
    let second = weight(5);
    let mut malformed = weight(9);
    malformed.data = Arc::<[u8]>::from(&malformed.data[..malformed.data.len() - 1]).into();
    let input = [1.0; 256];
    let mut scratch = MatvecScratch::with_pinned_backend(MatvecBackend::Auto);
    #[cfg(feature = "parallel-rows")]
    scratch.set_row_parallelism(RowParallelism::new(4)?);
    let mut outputs = std::array::from_fn::<_, 3, _>(|_| vec![f32::NAN]);
    {
        let mut prepared = scratch.prepare(&first, &input)?;
        assert!(
            prepared
                .matvec_pq2_siblings_into(&[&first, &second, &malformed], &mut outputs.each_mut())
                .is_err()
        );
    }
    assert!(outputs.iter().all(Vec::is_empty));
    assert_eq!(scratch.statistics().grouped_row_dispatches, 0);
    assert_eq!(scratch.statistics().matrix_successes, 0);
    assert!(!scratch.diagnostics().dispatched);

    let valid = weight(9);
    let mut invalid_scale = weight(5);
    let mut bytes = invalid_scale.data.to_vec();
    bytes[..2].copy_from_slice(&0x7c00_u16.to_le_bytes());
    invalid_scale.data = Arc::<[u8]>::from(bytes).into();
    let mut prepared = scratch.prepare(&first, &input)?;
    assert!(
        prepared
            .matvec_pq2_siblings_into(&[&first, &invalid_scale, &valid], &mut outputs.each_mut())
            .is_err()
    );
    assert!(outputs.iter().all(Vec::is_empty));
    // A failed group never poisons or requantizes the scoped activation.
    prepared.matvec_pq2_siblings_into(&[&first, &second, &valid], &mut outputs.each_mut())?;
    assert!(outputs.iter().all(|output| !output.is_empty()));
    assert_eq!(scratch.preparation_count(), 2);
    assert_eq!(scratch.statistics().matrix_successes, 3);

    #[cfg(feature = "parallel-rows")]
    let policy = RowParallelism::new(4)?;
    #[cfg(not(feature = "parallel-rows"))]
    let policy = RowParallelism::default();
    let data = [0_u8, 1, 2, 13, 4, 5, 6, 7];
    let later = [20_u8, 25, 22, 23, 24];
    let mut first_output = vec![0.0; data.len()];
    let mut second_output = vec![0.0; later.len()];
    let error = policy.fill_siblings(
        &[&data, &later],
        1,
        &mut [&mut first_output, &mut second_output],
        |bytes, values| {
            for (&byte, value) in bytes.iter().zip(values) {
                if byte == 13 || byte == 25 {
                    return Err(invalid(format!("rejected row {byte}")));
                }
                *value = f32::from(byte);
            }
            Ok(())
        },
    );
    assert_eq!(
        error.map_err(|error| error.to_string()),
        Err("rejected row 13".to_owned())
    );
    Ok(())
}

fn bf16_weight(rows: usize) -> Tensor {
    let bytes = [0x3f80_u16, 0xc000, 0x8000, 0x0080, 0x3eaa, 0x4000, 0x0001]
        .into_iter()
        .cycle()
        .take(256 * rows)
        .flat_map(u16::to_le_bytes)
        .collect::<Vec<_>>();
    Tensor {
        name: "bf16-sibling-control".to_owned(),
        dimensions: vec![256, rows],
        kind: 30,
        data: Arc::<[u8]>::from(bytes).into(),
    }
}

#[test]
fn bf16_siblings_reject_geometry_and_arithmetic_atomically_then_recover() -> Result<()> {
    let first = bf16_weight(17);
    let second = bf16_weight(5);
    let mut malformed = bf16_weight(9);
    malformed.data = Arc::<[u8]>::from(&malformed.data[..malformed.data.len() - 1]).into();
    let input = [1.0; 256];
    let mut scratch = MatvecScratch::with_pinned_backend(MatvecBackend::Auto);
    #[cfg(feature = "parallel-rows")]
    scratch.set_row_parallelism(RowParallelism::new(4)?);
    let mut outputs = std::array::from_fn::<_, 3, _>(|_| vec![f32::NAN]);
    {
        let mut prepared = scratch.prepare(&first, &input)?;
        assert!(
            prepared
                .matvec_bf16_siblings_into(&[&first, &second, &malformed], &mut outputs.each_mut())
                .is_err()
        );
    }
    assert!(outputs.iter().all(Vec::is_empty));
    assert_eq!(scratch.statistics().grouped_row_dispatches, 0);
    assert_eq!(scratch.statistics().matrix_successes, 0);
    assert!(!scratch.diagnostics().dispatched);

    let valid = bf16_weight(9);
    let mut invalid_value = bf16_weight(5);
    let mut bytes = invalid_value.data.to_vec();
    // Late row and column: every preceding output is still provisional.
    let end = bytes.len();
    bytes[end - 2..].copy_from_slice(&0x7f80_u16.to_le_bytes());
    invalid_value.data = Arc::<[u8]>::from(bytes).into();
    let mut prepared = scratch.prepare(&first, &input)?;
    assert!(
        prepared
            .matvec_bf16_siblings_into(&[&first, &invalid_value, &valid], &mut outputs.each_mut())
            .is_err()
    );
    assert!(outputs.iter().all(Vec::is_empty));
    prepared.matvec_bf16_siblings_into(&[&first, &second, &valid], &mut outputs.each_mut())?;
    assert!(outputs.iter().all(|output| !output.is_empty()));
    assert_eq!(scratch.preparation_count(), 2);
    assert_eq!(scratch.statistics().matrix_successes, 3);
    Ok(())
}

fn varied_pq2_weight(columns: usize, rows: usize, seed: usize) -> Tensor {
    let blocks = columns / 128;
    let scales: [u16; 8] = [
        0x3555, 0x3c01, 0xb787, 0x3800, 0xb400, 0x0000, 0x8000, 0x3a00,
    ];
    let mut data = Vec::with_capacity(blocks * rows * 34);
    for index in 0..blocks * rows {
        let scale = scales[(index + seed) % scales.len()];
        data.extend_from_slice(&scale.to_le_bytes());
        for byte in 0..32_usize {
            let mixed = (byte * 31 + index * 17 + seed * 13) % 256;
            data.push(mixed.to_le_bytes()[0]);
        }
    }
    Tensor {
        name: "pq2-sibling-varied".to_owned(),
        dimensions: vec![columns, rows],
        kind: super::super::PQ2,
        data: Arc::<[u8]>::from(data).into(),
    }
}

#[test]
fn pq2_siblings_sixteen_row_batching_matches_scalar_and_clears_atomically() -> Result<()> {
    let columns = 512;
    let input: Vec<f32> = (0..columns)
        .map(|lane| [0.0, -0.0, 1.5, -3.25, 63.5, -128.0, 127.0][lane % 7])
        .collect();
    let siblings = [
        varied_pq2_weight(columns, 68, 1),
        varied_pq2_weight(columns, 133, 2),
        varied_pq2_weight(columns, 65, 3),
        varied_pq2_weight(columns, 17, 4),
    ];

    for pinned in [false, true] {
        let mut scalar = if pinned {
            MatvecScratch::with_pinned_backend(MatvecBackend::Scalar)
        } else {
            MatvecScratch::with_backend(MatvecBackend::Scalar)
        };
        let expected: Vec<Vec<u32>> = siblings
            .iter()
            .map(|tensor| {
                let mut out = Vec::new();
                super::super::matvec_into(tensor, &input, &mut out, &mut scalar)?;
                Ok(out.into_iter().map(f32::to_bits).collect())
            })
            .collect::<Result<_>>()?;

        let worker_counts: &[usize] = if cfg!(feature = "parallel-rows") {
            &[1, 4]
        } else {
            &[1]
        };
        for &workers in worker_counts {
            let mut scratch = if pinned {
                MatvecScratch::with_pinned_backend(MatvecBackend::ExperimentalNeon)
            } else {
                MatvecScratch::with_backend(MatvecBackend::ExperimentalNeon)
            };
            scratch.set_row_parallelism(RowParallelism::new(workers)?);

            let mut outputs = std::array::from_fn::<_, 4, _>(|_| Vec::new());
            {
                let mut prepared = scratch.prepare(&siblings[0], &input)?;
                prepared.matvec_pq2_siblings_into(
                    &[&siblings[0], &siblings[1], &siblings[2], &siblings[3]],
                    &mut outputs.each_mut(),
                )?;
            }
            for (actual, exp) in outputs.iter().zip(&expected) {
                let bits: Vec<u32> = actual.iter().map(|v| v.to_bits()).collect();
                assert_eq!(&bits, exp);
            }

            // Inject nonfinite scale inside a 16-row batch (row 10 of sibling 1).
            let mut bad = siblings[1].clone();
            let row_bytes = (columns / 128) * 34;
            let mut bytes = bad.data.to_vec();
            let bad_offset = 10 * row_bytes + 34;
            bytes[bad_offset..bad_offset + 2].copy_from_slice(&0x7c01_u16.to_le_bytes());
            bad.data = Arc::<[u8]>::from(bytes).into();
            {
                let mut prepared = scratch.prepare(&siblings[0], &input)?;
                let [o0, o1, o2, _] = &mut outputs;
                assert!(
                    prepared
                        .matvec_pq2_siblings_into(
                            &[&siblings[0], &bad, &siblings[2]],
                            &mut [o0, o1, o2],
                        )
                        .is_err()
                );
            }
            assert!(outputs[..3].iter().all(Vec::is_empty));
            assert!(!scratch.diagnostics().succeeded);

            // Subsequent valid sibling call recovers cleanly.
            let mut recovered = std::array::from_fn::<_, 3, _>(|_| Vec::new());
            {
                let mut prepared = scratch.prepare(&siblings[0], &input)?;
                prepared.matvec_pq2_siblings_into(
                    &[&siblings[0], &siblings[1], &siblings[2]],
                    &mut recovered.each_mut(),
                )?;
            }
            assert!(scratch.diagnostics().succeeded);
            for (actual, exp) in recovered.iter().zip(&expected[..3]) {
                let bits: Vec<u32> = actual.iter().map(|v| v.to_bits()).collect();
                assert_eq!(&bits, exp);
            }
        }
    }
    Ok(())
}
