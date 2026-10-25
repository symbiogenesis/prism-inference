//! Native PTQ validation and malformed packed-activation regressions.
#![forbid(unsafe_code)]
use std::sync::Arc;

use bonsai_prism_runtime::kernels::{packed_dot, validate_tensor};
use bonsai_prism_runtime::model::Tensor;

type TestResult = Result<(), Box<dyn std::error::Error + Send + Sync>>;

#[test]
fn every_native_ptq_byte_is_accepted_by_tensor_validation() -> TestResult {
    // Decode and packed-dot parity for the same complete byte domain live in
    // upstream_prism::codec_assertions alongside the original CUDA-host port.
    // Keep the independent validation entry point covered here.
    for position in 0..26 {
        for byte in 0..=255 {
            let mut packed = [0; 28];
            packed[26..].copy_from_slice(&0x3c00_u16.to_le_bytes());
            packed[position] = byte;
            let tensor = Tensor {
                name: "native-ptq".to_owned(),
                kind: 143,
                dimensions: vec![128],
                data: Arc::from(packed),
            };
            validate_tensor(&tensor)?;
        }
    }
    Ok(())
}

#[test]
fn packed_dot_rejects_wrong_extents_nonfinite_scales_and_incomplete_blocks() {
    let mut weights = [0; 28];
    weights[26..].copy_from_slice(&0x3c00_u16.to_le_bytes());
    let activations = [0; 136];
    assert!(packed_dot(143, 128, &weights[..27], &activations).is_err());
    assert!(packed_dot(143, 128, &weights, &activations[..135]).is_err());
    assert!(packed_dot(143, 127, &weights, &activations).is_err());
    assert!(packed_dot(143, 0, &[], &[]).is_err());
    weights[26..].copy_from_slice(&0x7c00_u16.to_le_bytes());
    assert!(packed_dot(143, 128, &weights, &activations).is_err());
    weights[26..].copy_from_slice(&0x3c00_u16.to_le_bytes());
    let mut invalid = activations;
    invalid[..2].copy_from_slice(&0x7e00_u16.to_le_bytes());
    assert!(packed_dot(143, 128, &weights, &invalid).is_err());
}
