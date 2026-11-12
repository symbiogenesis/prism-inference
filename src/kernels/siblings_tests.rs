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
        data: Arc::from(block.repeat(2 * rows)),
    }
}

#[test]
fn cpu_sibling_projection_rejects_late_geometry_and_recovers() -> Result<()> {
    let first = weight(17);
    let second = weight(5);
    let mut malformed = weight(9);
    malformed.data = Arc::from(&malformed.data[..malformed.data.len() - 1]);
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
    invalid_scale.data = Arc::from(bytes);
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
