use std::sync::Arc;

use super::*;
use crate::kernels::{MatvecBackend, MatvecScratch, matvec_into};

fn tensor(columns: usize, rows: usize) -> Result<Tensor> {
    let mut data = Vec::new();
    for index in 0..columns / 128 * rows {
        // Include signed zero, unequal scales and arbitrary fourth symbols.
        data.extend_from_slice(
            &[0x0000_u16, 0x8000, 0x3555, 0xb800, 0x3c00][index % 5].to_le_bytes(),
        );
        for byte in 0..32 {
            data.push(u8::try_from((index * 19 + byte * 73 + 47) % 256)?);
        }
    }
    Ok(Tensor {
        name: "pq2-lut-control".into(),
        dimensions: vec![columns, rows],
        kind: PQ2,
        data: Arc::from(data),
    })
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

#[test]
fn lut_projection_preserves_quantizers_accumulation_and_activation_scope() -> Result<()> {
    let workers = if cfg!(feature = "parallel-rows") {
        4
    } else {
        1
    };
    let mut executor = Pq2LutExecutor::new(5120, RowParallelism::new(workers)?)?;
    for columns in [128, 256, 384, 512, 5120] {
        for pinned in [false, true] {
            let tensor = tensor(columns, 9)?;
            let make_scratch = || {
                if pinned {
                    MatvecScratch::with_pinned_backend(MatvecBackend::Scalar)
                } else {
                    MatvecScratch::with_backend(MatvecBackend::Scalar)
                }
            };
            let mut reference = make_scratch();
            let mut actual = make_scratch();
            let mut input: Vec<_> = (0..columns)
                .map(|index| [0.0, -0.0, 1.5, -127.0, -128.0, 63.5, 17.25][index % 7])
                .collect();
            let (mut expected, mut output) = (Vec::new(), Vec::new());
            for _ in 0..2 {
                matvec_into(&tensor, &input, &mut expected, &mut reference)?;
                let before = executor.preparations;
                actual.prepare(&tensor, &input)?.matvec_with_executor(
                    &tensor,
                    &mut output,
                    &mut executor,
                )?;
                assert_eq!(bits(&output), bits(&expected));
                // Each table is built once for all nine rows, never per row.
                assert_eq!(executor.preparations - before, columns / 128);
                assert!(executor.tables.is_empty());
                assert!(executor.row_errors.is_empty());
                // Reuse the same input storage with different quantized codes.
                input.rotate_left(1);
            }
        }
    }
    Ok(())
}

#[test]
fn lut_projection_checks_extents_scales_limits_and_late_row_failure() -> Result<()> {
    assert!(Pq2LutExecutor::new(0, RowParallelism::new(1)?).is_err());
    assert!(Pq2LutExecutor::new(129, RowParallelism::new(1)?).is_err());
    let mut executor = Pq2LutExecutor::new(384, RowParallelism::new(1)?)?;
    assert!(!executor.supports(&tensor(512, 1)?));
    let mut activation = Q8KBlock {
        scale: -0.125,
        values: [-128; 256],
        sums: [0; 16],
    };
    let valid = tensor(256, 3)?;
    let mut output = vec![42.0];
    assert!(executor.project(&valid, &[], false, &mut output).is_err());
    assert!(output.is_empty());
    activation.scale = f32::NAN;
    assert!(
        executor
            .project(&valid, std::slice::from_ref(&activation), true, &mut output)
            .is_err()
    );
    assert!(output.is_empty());
    assert!(executor.tables.is_empty());
    activation.scale = -0.125;
    let mut invalid = valid.clone();
    let bytes = Arc::make_mut(&mut invalid.data);
    let row_bytes = 2 * 34;
    bytes[row_bytes * 2..row_bytes * 2 + 2].copy_from_slice(&0x7c00_u16.to_le_bytes());
    assert!(
        executor
            .project(
                &invalid,
                std::slice::from_ref(&activation),
                false,
                &mut output
            )
            .is_err()
    );
    assert!(output.is_empty());
    assert!(executor.tables.is_empty());
    executor.project(
        &valid,
        std::slice::from_ref(&activation),
        false,
        &mut output,
    )?;
    let expected = valid
        .data
        .chunks_exact(row_bytes)
        .map(|row| super::super::dot_pq2::<false>(row, std::slice::from_ref(&activation)))
        .collect::<Result<Vec<_>>>()?;
    assert_eq!(bits(&output), bits(&expected));
    let odd = tensor(128, 1)?;
    let mut q8 = Vec::new();
    for scale in [0.0, -0.0, 0.125, -0.25] {
        q8.push(Q8Block {
            scale,
            sum: 0.0,
            values: [-128; 32],
        });
    }
    executor.project_q8(&odd, &q8, true, &mut output)?;
    assert_eq!(
        bits(&output),
        bits(&[super::super::dot_q8_groups(PQ2, &odd.data, &q8)?])
    );
    q8[3].scale = f32::INFINITY;
    assert!(executor.project_q8(&odd, &q8, false, &mut output).is_err());
    assert!(output.is_empty());
    assert!(executor.tables.is_empty());
    Ok(())
}

#[test]
fn lut_tables_cover_every_pq2_symbol_and_integer_boundary() {
    for value in [-128, -127, 0, 1, 127] {
        let activations = [value; 128];
        let table = Tl1BlockLut::from_contiguous_activations(&activations);
        for byte in 0_u8..=255 {
            let packed = [byte; 32];
            let digit_sum: i32 = (0..4)
                .map(|digit| i32::from((byte >> (2 * digit)) & 3) - 1)
                .sum();
            let expected = digit_sum * i32::from(value) * 32;
            assert_eq!(table.dot(&packed), expected);
            assert_eq!(table.dot_groups(&packed), [expected / 4; 4]);
        }
    }
}
