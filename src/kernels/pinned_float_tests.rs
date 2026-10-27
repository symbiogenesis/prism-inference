//! Independent native controls for the aligned pinned floating projection route.

use std::io::{Cursor, Read};
use std::sync::Arc;

use serde::Deserialize;
use sha2::{Digest, Sha256};

use super::{
    BF16, F16, F32, MatvecBackend, MatvecScratch, decode_row_into, matvec_into, packed_dot,
};
use crate::Result;
use crate::model::Tensor;

#[derive(Deserialize)]
struct Capture {
    runtime_commit: String,
    ggml_cpu_sha256: String,
    fixture_bytes: usize,
    fixture_sha256: String,
    records: Vec<Record>,
}

#[derive(Deserialize)]
struct Record {
    kind: u32,
    count: usize,
    native_bits: u32,
    generic_bits: u32,
}

fn read_u32(input: &mut impl Read) -> Result<u32> {
    let mut bytes = [0; 4];
    input.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

fn tensor(kind: u32, count: usize, bytes: Vec<u8>) -> Tensor {
    Tensor {
        name: "native-floating-control".to_owned(),
        dimensions: vec![count, 1],
        kind,
        data: Arc::from(bytes),
    }
}

#[test]
fn aligned_pinned_float_projections_match_native_without_changing_generic_or_bf16() -> Result<()> {
    let capture: Capture = serde_json::from_str(include_str!(
        "../../tests/fixtures/swift-float-projection-parity.json"
    ))?;
    let bytes = include_bytes!("../../tests/fixtures/swift-float-projection-parity.bin");
    assert_eq!(
        capture.runtime_commit,
        "73bfe6cc485554822f80c473a3c0c4f25e80f41b"
    );
    assert_eq!(
        capture.ggml_cpu_sha256,
        "701b354a38b25e5c6dc57a1559f71cd26b2d0f8d69c1576a0e77a59a4273d2ab"
    );
    assert_eq!(bytes.len(), capture.fixture_bytes);
    let hash: String = Sha256::digest(bytes)
        .into_iter()
        .flat_map(|byte| [byte >> 4, byte & 15])
        .map(|nibble| char::from(b"0123456789abcdef"[usize::from(nibble)]))
        .collect();
    assert_eq!(hash, capture.fixture_sha256);
    assert_eq!(
        hash,
        "5a00f4c15e67ba7dd49a4e38c833e07648e7f617e0c2fd6d57078f3e55e9c263"
    );
    assert_eq!(capture.records.len(), 12);
    let mut reader = Cursor::new(bytes.as_slice());
    let mut different = 0;
    for record in capture.records {
        assert_eq!(read_u32(&mut reader)?, record.kind);
        assert_eq!(usize::try_from(read_u32(&mut reader)?)?, record.count);
        assert_eq!(read_u32(&mut reader)?, record.native_bits);
        assert!(record.count.is_multiple_of(16));
        let width = if record.kind == F32 { 4 } else { 2 };
        let mut weights = vec![0; record.count * width];
        let mut activations = vec![0; weights.len()];
        reader.read_exact(&mut weights)?;
        reader.read_exact(&mut activations)?;
        let generic = packed_dot(record.kind, record.count, &weights, &activations)?;
        assert_eq!(generic.to_bits(), record.generic_bits);
        let weights = tensor(record.kind, record.count, weights);
        let mut input = Vec::new();
        decode_row_into(record.kind, record.count, &activations, &mut input)?;
        for backend in [MatvecBackend::Scalar, MatvecBackend::Auto] {
            let mut output = Vec::new();
            matvec_into(
                &weights,
                &input,
                &mut output,
                &mut MatvecScratch::with_backend(backend),
            )?;
            assert_eq!(output[0].to_bits(), record.generic_bits);
            matvec_into(
                &weights,
                &input,
                &mut output,
                &mut MatvecScratch::with_pinned_backend(backend),
            )?;
            assert_eq!(
                output[0].to_bits(),
                record.native_bits,
                "kind {}, width {}",
                record.kind,
                record.count
            );
        }
        if record.kind == BF16 {
            assert_eq!(record.native_bits, record.generic_bits);
        } else if record.native_bits != record.generic_bits {
            different += 1;
        }
    }
    assert_eq!(
        different, 3,
        "native fixtures distinguish the repaired route"
    );
    assert_eq!(usize::try_from(reader.position())?, bytes.len());
    Ok(())
}

#[test]
fn unqualified_float_projection_tails_keep_the_previous_arithmetic() -> Result<()> {
    for kind in [F32, F16, BF16] {
        for count in [1, 3, 15, 17, 31, 33] {
            let input = (0..count)
                .map(|index| if index % 2 == 0 { 0.75 } else { -0.5 })
                .collect::<Vec<_>>();
            let values = (0..count)
                .map(|index| if index % 3 == 0 { 2.5 } else { -1.25 })
                .collect::<Vec<_>>();
            let weights = tensor(kind, count, super::quantize_row(kind, &values)?);
            let mut generic = Vec::new();
            let mut pinned = Vec::new();
            matvec_into(
                &weights,
                &input,
                &mut generic,
                &mut MatvecScratch::default(),
            )?;
            matvec_into(
                &weights,
                &input,
                &mut pinned,
                &mut MatvecScratch::with_pinned_backend(MatvecBackend::Auto),
            )?;
            assert_eq!(
                generic
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                pinned
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>()
            );
        }
    }
    Ok(())
}
