//! Native controls for pinned PTQ1 scale and inverse preparation arithmetic.

use std::sync::Arc;

use serde::Deserialize;
use sha2::{Digest, Sha256};

use super::{
    MatvecBackend, MatvecScratch, PQ2, PTQ1, Q8_0, Q8Block, half_bits, q8_with_contract,
    q8_with_rounding, quantize_row,
};
use crate::Result;
use crate::model::Tensor;

#[derive(Deserialize)]
struct Capture {
    runtime_commit: String,
    ggml_cpu_sha256: String,
    fixture_bytes: usize,
    fixture_sha256: String,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    input_bits: Vec<u32>,
}

fn capture() -> Result<Capture> {
    serde_json::from_str(include_str!(
        "../../tests/fixtures/swift-ptq1-q8-scale-parity.json"
    ))
    .map_err(Into::into)
}

fn packed(block: &Q8Block) -> Vec<u8> {
    half_bits(block.scale)
        .to_le_bytes()
        .into_iter()
        .chain(block.values.iter().map(|value| value.to_ne_bytes()[0]))
        .collect()
}

#[test]
fn pinned_ptq1_scale_and_inverse_match_native_halfway_and_zero_controls() -> Result<()> {
    let capture = capture()?;
    let fixture = include_bytes!("../../tests/fixtures/swift-ptq1-q8-scale-parity.bin");
    let hash: String = Sha256::digest(fixture)
        .into_iter()
        .flat_map(|byte| [byte >> 4, byte & 15])
        .map(|nibble| char::from(b"0123456789abcdef"[usize::from(nibble)]))
        .collect();
    assert_eq!(
        capture.runtime_commit,
        "73bfe6cc485554822f80c473a3c0c4f25e80f41b"
    );
    assert_eq!(
        capture.ggml_cpu_sha256,
        "701b354a38b25e5c6dc57a1559f71cd26b2d0f8d69c1576a0e77a59a4273d2ab"
    );
    assert_eq!(capture.fixture_bytes, fixture.len());
    assert_eq!(capture.fixture_sha256, hash);
    assert_eq!(
        hash,
        "cda434b51d60bbdcd417492a4af77f3ef6cb5eef6390826abb6609b8a3578dee"
    );
    assert_eq!(capture.cases.len(), 4);
    let mut blocks = Vec::new();
    for (case, expected) in capture.cases.iter().zip(fixture.as_chunks::<34>().0) {
        let input = case
            .input_bits
            .iter()
            .copied()
            .map(f32::from_bits)
            .collect::<Vec<_>>();
        assert_eq!(input.len(), 32);
        q8_with_contract::<true, true>(&input, &mut blocks)?;
        assert_eq!(blocks.len(), 1);
        assert_eq!(packed(&blocks[0]), expected.as_slice(), "{}", case.name);
    }
    assert_eq!(i8::from_ne_bytes([fixture[3]]), -52);
    assert_eq!(i8::from_ne_bytes([fixture[34 + 3]]), -77);
    Ok(())
}

fn weights(kind: u32) -> Result<Tensor> {
    Ok(Tensor {
        name: "prepared-contract-control".to_owned(),
        dimensions: vec![128, 1],
        kind,
        data: Arc::from(quantize_row(kind, &[1.0; 128])?),
    })
}

#[test]
fn corrected_ptq1_preparation_preserves_prior_pq2_and_generic_contracts() -> Result<()> {
    let capture = capture()?;
    let block = capture.cases[0]
        .input_bits
        .iter()
        .copied()
        .map(f32::from_bits)
        .collect::<Vec<_>>();
    let input = block.repeat(4);
    let ptq1 = weights(PTQ1)?;
    let pq2 = weights(PQ2)?;
    let q8 = weights(Q8_0)?;
    let mut scratch = MatvecScratch::with_pinned_backend(MatvecBackend::Scalar);
    {
        let mut prepared = scratch.prepare(&ptq1, &input)?;
        assert!(prepared.is_compatible(&ptq1)?);
        assert!(!prepared.is_compatible(&pq2)?);
        assert!(!prepared.is_compatible(&q8)?);
        assert_eq!(prepared.scratch.q8[0].values[1], -52);
        let mut output = vec![99.0];
        assert!(prepared.matvec_into(&pq2, &mut output).is_err());
        let empty: [f32; 0] = [];
        assert_eq!(output, empty);
    }
    {
        let prepared = scratch.prepare(&pq2, &input)?;
        assert!(prepared.is_compatible(&q8)?);
        assert!(!prepared.is_compatible(&ptq1)?);
        assert_eq!(prepared.scratch.q8[0].values[1], -51);
    }
    let mut prior = Vec::new();
    q8_with_rounding::<true>(&input, &mut prior)?;
    assert_eq!(packed(&scratch.q8[0]), packed(&prior[0]));
    let mut generic = MatvecScratch::default();
    let prepared = generic.prepare(&ptq1, &input)?;
    assert!(prepared.is_compatible(&pq2)?);
    assert!(prepared.is_compatible(&q8)?);
    assert_eq!(prepared.scratch.q8[0].values[1], -51);
    Ok(())
}
