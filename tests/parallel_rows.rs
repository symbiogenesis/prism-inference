#![forbid(unsafe_code)]
//! Exact arithmetic and failure controls for explicitly budgeted projection rows.

use bonsai_prism_runtime::Result;
use bonsai_prism_runtime::kernels::{MAX_ROW_WORKERS, RowParallelism};

#[test]
fn row_worker_policy_is_explicit_and_bounded() -> Result<()> {
    assert_eq!(RowParallelism::default().workers(), 1);
    assert_eq!(RowParallelism::new(1)?.workers(), 1);
    assert!(RowParallelism::new(0).is_err());
    assert!(RowParallelism::new(MAX_ROW_WORKERS + 1).is_err());
    #[cfg(feature = "parallel-rows")]
    assert_eq!(RowParallelism::new(2)?.workers(), 2);
    #[cfg(not(feature = "parallel-rows"))]
    assert!(RowParallelism::new(2).is_err());
    Ok(())
}

#[cfg(feature = "parallel-rows")]
mod enabled {
    use std::sync::Arc;

    use bonsai_prism_runtime::kernels::{MatvecBackend, MatvecScratch, matvec_into};
    use bonsai_prism_runtime::model::Tensor;
    use bonsai_ternary::pack_ptq;

    use super::{Result, RowParallelism};

    fn packed(kind: u32, columns: usize, rows: usize) -> Result<Tensor> {
        let mut data = Vec::new();
        for row in 0..rows {
            for group in 0..columns / 128 {
                let mut block = [0_u8; 34];
                let scale = [0x3400_u16, 0xb400, 0, 0x8000][(row + group) % 4];
                block[..2].copy_from_slice(&scale.to_le_bytes());
                for index in 0..128 {
                    let symbols = if kind == 143 { 3 } else { 4 };
                    let code = u8::try_from((row * 3 + group * 5 + index / 7 + index) % symbols)?;
                    block[2 + index / 4] |= code << (2 * (index % 4));
                }
                if kind == 143 {
                    let mut encoded = pack_ptq(&{
                        let mut nonzero = block;
                        nonzero[..2].copy_from_slice(&0x3400_u16.to_le_bytes());
                        nonzero
                    })?;
                    encoded[26..].copy_from_slice(&scale.to_le_bytes());
                    data.extend_from_slice(&encoded);
                } else {
                    data.extend_from_slice(&block);
                }
            }
        }
        Ok(Tensor {
            name: "parallel-control".into(),
            dimensions: vec![columns, rows],
            kind,
            data: Arc::from(data),
        })
    }

    fn assert_bits(first: &[f32], second: &[f32]) {
        assert_eq!(
            first
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            second
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn parallel_rows_preserve_packed_arithmetic_and_prepared_siblings() -> Result<()> {
        let policy = RowParallelism::new(3)?;
        for backend in [MatvecBackend::Scalar, MatvecBackend::Auto] {
            // Actual Swift input widths plus odd-group fallback and awkward rows.
            for (kind, width, rows) in [
                (142, 5120, 17),
                (142, 17408, 7),
                (143, 6144, 19),
                (142, 384, 11),
            ] {
                let tensor = packed(kind, width, rows)?;
                let owner = Arc::clone(&tensor.data);
                let input: Vec<_> = (0..width)
                    .map(|index| [0.0, -0.0, 1.5, -2.5, 63.5, -127.0][index % 6])
                    .collect();
                let mut serial = MatvecScratch::with_backend(backend);
                let mut parallel = MatvecScratch::with_backend(backend);
                parallel.set_row_parallelism(policy.clone());
                let mut expected = Vec::new();
                let mut actual = Vec::new();
                matvec_into(&tensor, &input, &mut expected, &mut serial)?;
                let mut prepared = parallel.prepare(&tensor, &input)?;
                prepared.matvec_into(&tensor, &mut actual)?;
                assert_bits(&actual, &expected);
                let first_pointer = actual.as_ptr();
                prepared.matvec_into(&tensor, &mut actual)?;
                assert_bits(&actual, &expected);
                assert_eq!(actual.as_ptr(), first_pointer);
                assert_eq!(parallel.row_workers(), 3);
                assert_eq!(parallel.statistics().parallel_row_operations, 2);
                assert_eq!(parallel.selected_backend(), serial.selected_backend());
                assert!(Arc::ptr_eq(&owner, &tensor.data));
                assert_eq!(Arc::strong_count(&tensor.data), 2);
            }
        }
        Ok(())
    }

    #[test]
    fn parallel_rows_clear_failed_output_and_recover_with_serial_float_fallback() -> Result<()> {
        let mut parallel = MatvecScratch::with_backend(MatvecBackend::Auto);
        parallel.set_row_parallelism(RowParallelism::new(2)?);
        let tensor = packed(143, 128, 9)?;
        let input = vec![0.75; 128];
        let mut output = vec![f32::NAN];
        let mut malformed = tensor.clone();
        let mut data = malformed.data.to_vec();
        data[5 * 28 + 24] = 255;
        malformed.data = Arc::from(data.clone());
        // Native PTQ aliases are decodable; only the lossless wire exporter
        // requires canonical bytes. Scheduling preserves their current meaning.
        let mut alias_reference = Vec::new();
        matvec_into(
            &malformed,
            &input,
            &mut alias_reference,
            &mut MatvecScratch::default(),
        )?;
        matvec_into(&malformed, &input, &mut output, &mut parallel)?;
        assert_bits(&output, &alias_reference);
        data[5 * 28 + 26..5 * 28 + 28].copy_from_slice(&0x7c00_u16.to_le_bytes());
        malformed.data = Arc::from(data);
        assert!(matvec_into(&malformed, &input, &mut output, &mut parallel).is_err());
        assert_eq!(output, Vec::<f32>::new());
        assert!(!parallel.diagnostics().succeeded);
        matvec_into(&tensor, &input, &mut output, &mut parallel)?;
        assert_eq!(output.len(), 9);
        let mut invalid_scale = packed(142, 256, 9)?;
        let mut bytes = invalid_scale.data.to_vec();
        bytes[8 * 68..8 * 68 + 2].copy_from_slice(&0x7e00_u16.to_le_bytes());
        invalid_scale.data = Arc::from(bytes);
        assert!(matvec_into(&invalid_scale, &[0.75; 256], &mut output, &mut parallel).is_err());
        assert_eq!(output, Vec::<f32>::new());
        let mut truncated = tensor.clone();
        truncated.data = Arc::from(&tensor.data[..tensor.data.len() - 1]);
        assert!(matvec_into(&truncated, &input, &mut output, &mut parallel).is_err());
        assert_eq!(output, Vec::<f32>::new());
        let dispatched = parallel.statistics().parallel_row_operations;
        let floats = Tensor {
            name: "floating-fallback".into(),
            dimensions: vec![2, 2],
            kind: 0,
            data: Arc::from(
                [1.0_f32, -0.0, -0.25, 0.75]
                    .into_iter()
                    .flat_map(f32::to_le_bytes)
                    .collect::<Vec<_>>(),
            ),
        };
        let mut expected = Vec::new();
        matvec_into(
            &floats,
            &[1.5, -2.0],
            &mut expected,
            &mut MatvecScratch::default(),
        )?;
        matvec_into(&floats, &[1.5, -2.0], &mut output, &mut parallel)?;
        assert_bits(&output, &expected);
        assert_eq!(parallel.statistics().parallel_row_operations, dispatched);
        assert!(matvec_into(&tensor, &[f32::NAN; 128], &mut output, &mut parallel).is_err());
        assert_eq!(output, Vec::<f32>::new());
        Ok(())
    }
}
