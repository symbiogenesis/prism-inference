//! Scoped finite-input proof controls against the original quantizer entry points.

use std::sync::Arc;

use super::{
    MatvecBackend, MatvecScratch, PQ2, PTQ1, Q8_0, q8_with_contract, q8k_with_rounding,
    quantize_row,
};
use crate::Result;
use crate::model::Tensor;

fn tensor(kind: u32, columns: usize) -> Result<Tensor> {
    Ok(Tensor {
        name: "finite-activation-proof".to_owned(),
        dimensions: vec![columns, 1],
        kind,
        data: Arc::from(quantize_row(kind, &vec![1.0; columns])?),
    })
}

fn same_q8<const PINNED: bool, const NATIVE_SCALE: bool>(
    scratch: &MatvecScratch,
    input: &[f32],
) -> Result<()> {
    let mut original = Vec::new();
    q8_with_contract::<PINNED, NATIVE_SCALE>(input, &mut original)?;
    assert_eq!(scratch.q8.len(), original.len());
    for (actual, expected) in scratch.q8.iter().zip(original) {
        assert_eq!(actual.scale.to_bits(), expected.scale.to_bits());
        assert_eq!(actual.sum.to_bits(), expected.sum.to_bits());
        assert_eq!(actual.values, expected.values);
    }
    Ok(())
}

fn same_q8k<const PINNED: bool>(scratch: &MatvecScratch, input: &[f32]) -> Result<()> {
    let mut original = Vec::new();
    q8k_with_rounding::<PINNED>(input, &mut original)?;
    assert_eq!(scratch.q8k.len(), original.len());
    for (actual, expected) in scratch.q8k.iter().zip(original) {
        assert_eq!(actual.scale.to_bits(), expected.scale.to_bits());
        assert_eq!(actual.values, expected.values);
        assert_eq!(actual.sums, expected.sums);
    }
    Ok(())
}

#[test]
fn prepared_quantizers_reuse_finite_proof_with_original_code_bits() -> Result<()> {
    let mut input = (0_u16..5120)
        .map(|index| (f32::from(index % 509) - 254.0) / 2.0)
        .collect::<Vec<_>>();
    let edges = [
        -0.0,
        0.0,
        f32::from_bits(1),
        -f32::from_bits(1),
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
        0.5,
        -0.5,
        127.0,
        -127.0,
    ];
    input[..edges.len()].copy_from_slice(&edges);
    for pinned in [false, true] {
        for backend in [MatvecBackend::Scalar, MatvecBackend::ExperimentalNeon] {
            let mut scratch = if pinned {
                MatvecScratch::with_pinned_backend(backend)
            } else {
                MatvecScratch::with_backend(backend)
            };
            for kind in [PQ2, PTQ1, Q8_0] {
                let weights = tensor(kind, input.len())?;
                scratch.prepare(&weights, &input)?;
                if kind == PQ2 {
                    if pinned {
                        same_q8k::<true>(&scratch, &input)?;
                    } else {
                        same_q8k::<false>(&scratch, &input)?;
                    }
                } else if !pinned {
                    same_q8::<false, false>(&scratch, &input)?;
                } else if kind == PTQ1 {
                    same_q8::<true, true>(&scratch, &input)?;
                } else {
                    same_q8::<true, false>(&scratch, &input)?;
                }
            }
        }
    }
    Ok(())
}

fn error(scratch: &mut MatvecScratch, tensor: &Tensor, input: &[f32]) -> String {
    match scratch.prepare(tensor, input) {
        Ok(_) => String::new(),
        Err(error) => error.to_string(),
    }
}

#[test]
fn prepared_finite_proof_preserves_preflight_precedence_and_storage() -> Result<()> {
    let weights = tensor(PQ2, 512)?;
    let mut scratch = MatvecScratch::with_pinned_backend(MatvecBackend::Scalar);
    scratch.prepare(&weights, &[127.0; 512])?;
    super::packed_pq2::prepare(&scratch.q8k, &mut scratch.pq2_consumption)?;
    assert!(!scratch.pq2_consumption.is_empty());
    let pointer = scratch.q8k.as_ptr();
    let mut mixed_failure = [f32::from_bits(1); 512];
    mixed_failure[511] = f32::NAN;
    let invalid_input = "matvec requires a matching finite activation vector";
    assert_eq!(error(&mut scratch, &weights, &mixed_failure), invalid_input);
    assert!(scratch.pq2_consumption.is_empty());
    // Preflight failure precedes quantizer clearing and scale overflow, just as
    // before proof reuse. No PreparedMatvec can expose the retained old codes.
    same_q8k::<true>(&scratch, &[127.0; 512])?;
    assert_eq!(scratch.q8k.as_ptr(), pointer);
    assert_eq!(error(&mut scratch, &weights, &[f32::NAN]), invalid_input);
    let mut malformed = weights.clone();
    malformed.dimensions.clear();
    assert_eq!(
        error(&mut scratch, &malformed, &mixed_failure),
        "tensor requires at least one dimension"
    );
    malformed.dimensions = vec![512, 1];
    malformed.kind = u32::MAX;
    assert_eq!(
        error(&mut scratch, &malformed, &mixed_failure),
        "unsupported runtime tensor type 4294967295"
    );
    mixed_failure[511] = f32::from_bits(1);
    assert_eq!(
        error(&mut scratch, &weights, &mixed_failure),
        "Q8_K activation scale overflow"
    );
    assert!(scratch.q8k.is_empty());
    assert_eq!(scratch.q8k.as_ptr(), pointer);
    scratch.prepare(&weights, &[-127.0; 512])?;
    same_q8k::<true>(&scratch, &[-127.0; 512])?;
    assert_eq!(scratch.q8k.as_ptr(), pointer);
    Ok(())
}

#[test]
fn prepared_finite_proof_retains_q8_overflow_and_external_validation() -> Result<()> {
    for pinned in [false, true] {
        for kind in [PTQ1, Q8_0] {
            let weights = tensor(kind, 256)?;
            let mut scratch = if pinned {
                MatvecScratch::with_pinned_backend(MatvecBackend::Scalar)
            } else {
                MatvecScratch::default()
            };
            scratch.prepare(&weights, &[127.0; 256])?;
            let pointer = scratch.q8.as_ptr();
            assert!(!error(&mut scratch, &weights, &[f32::MAX; 256]).is_empty());
            assert!(scratch.q8.is_empty());
            assert_eq!(scratch.q8.as_ptr(), pointer);
            assert!(
                scratch
                    .prepare_f32_executor(&weights, &[f32::NAN; 256])
                    .is_err()
            );
            scratch.prepare_f32_executor(&weights, &[127.0; 256])?;
            assert!(scratch.q8.is_empty());
            scratch.prepare(&weights, &[0.0; 256])?;
            assert_eq!(scratch.q8.as_ptr(), pointer);
            assert!(scratch.q8.iter().all(|block| block.values == [0; 32]));
        }
    }
    Ok(())
}
