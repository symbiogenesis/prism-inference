//! Native correctness observations for Prism IQ3 decoding and packed dots.

use bonsai_prism_runtime::backend::{Argument, CpuBackend, Observation};
use bonsai_prism_runtime::{Result, kernels, model::Tensor};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::sync::Arc;

const FIXTURE: &str = include_str!("fixtures/upstream-iq3.json");
const SCALES: [u16; 8] = [0, 0x8000, 1, 0x8001, 0x3c00, 0xbc00, 0x7bff, 0xfbff];

fn bytes(value: &Value) -> Result<Vec<u8>> {
    let text = value
        .as_str()
        .ok_or_else(|| std::io::Error::other("IQ3 fixture hex missing"))?;
    if !text.len().is_multiple_of(2) {
        return Err(std::io::Error::other("odd IQ3 fixture hex length").into());
    }
    text.as_bytes()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| Ok(u8::from_str_radix(std::str::from_utf8(pair)?, 16)?))
        .collect()
}

fn tensor(kind: u32, count: usize, bytes: Vec<u8>) -> Tensor {
    Tensor {
        name: "iq3-native".into(),
        kind,
        dimensions: vec![count],
        data: Arc::from(bytes),
    }
}

fn kind(record: &Value) -> Result<u32> {
    Ok(u32::try_from(record["TypeId"].as_u64().ok_or_else(
        || std::io::Error::other("IQ3 kind missing"),
    )?)?)
}

fn activation(seed: usize) -> Result<Vec<u8>> {
    let mut bytes = Vec::with_capacity(292);
    let d: f32 = match seed % 3 {
        0 => -0.03125,
        1 => 1.1,
        _ => 0.000_001,
    };
    bytes.extend(d.to_le_bytes());
    for index in 0..256 {
        bytes.push(u8::try_from((seed + index * 13) % 256)?);
    }
    for index in 0..16 {
        bytes.extend_from_slice(&(seed * 257 + index * 31).to_le_bytes()[..2]);
    }
    Ok(bytes)
}

#[test]
fn native_multiblock_decode_and_integer_dot_are_byte_exact() -> Result<()> {
    let fixture: Value = serde_json::from_str(FIXTURE)?;
    let cases = fixture["Cases"]
        .as_array()
        .ok_or_else(|| std::io::Error::other("IQ3 cases missing"))?;
    assert_eq!(cases.len(), 16);
    for case in cases {
        let kind = kind(case)?;
        let packed = bytes(&case["WeightsHex"])?;
        let weights = tensor(kind, 768, packed.clone());
        kernels::validate_tensor(&weights)?;
        let decoded = kernels::row(&weights, 0)?;
        assert_eq!(decoded.len(), 768);
        assert_eq!(
            decoded
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>(),
            bytes(&case["DecodedHex"])?,
            "decode kind {kind} scenario {}",
            case["Scenario"]
        );
        assert_eq!(
            kernels::packed_dot(kind, 768, &packed, &bytes(&case["ActivationsHex"])?)?
                .to_le_bytes()
                .as_slice(),
            bytes(&case["DotHex"])?,
            "dot kind {kind} scenario {}",
            case["Scenario"]
        );
    }
    Ok(())
}

#[test]
fn every_payload_position_and_byte_matches_native_decode_and_dot_hashes() -> Result<()> {
    let fixture: Value = serde_json::from_str(FIXTURE)?;
    let records = fixture["Arbitrary"]
        .as_array()
        .ok_or_else(|| std::io::Error::other("IQ3 hashes missing"))?;
    assert_eq!(records.len(), 2);
    for record in records {
        let kind = kind(record)?;
        let size = if kind == 18 { 98 } else { 110 };
        let mut decoded = Sha256::new();
        let mut dots = Sha256::new();
        for position in 2..size {
            for byte in 0..=255_u8 {
                let mut packed = vec![0; size];
                packed[..2]
                    .copy_from_slice(&SCALES[(usize::from(byte) + position) % 8].to_le_bytes());
                for (index, value) in packed.iter_mut().enumerate().skip(2) {
                    *value = u8::try_from((index * 37 + position * 17) % 256)?;
                }
                packed[position] = byte;
                let weights = tensor(kind, 256, packed.clone());
                kernels::validate_tensor(&weights)?;
                for value in kernels::row(&weights, 0)? {
                    decoded.update(value.to_le_bytes());
                }
                dots.update(
                    kernels::packed_dot(
                        kind,
                        256,
                        &packed,
                        &activation(usize::from(byte) + position)?,
                    )?
                    .to_le_bytes(),
                );
            }
        }
        assert_eq!(record["Blocks"], (size - 2) * 256);
        assert_eq!(
            decoded.finalize().as_slice(),
            bytes(&record["DecodedSha256"])?,
            "decode hash kind {kind}"
        );
        assert_eq!(
            dots.finalize().as_slice(),
            bytes(&record["DotSha256"])?,
            "dot hash kind {kind}"
        );
    }
    Ok(())
}

#[test]
fn checked_rows_prepared_matvec_and_explicit_encoder_limits() -> Result<()> {
    let fixture: Value = serde_json::from_str(FIXTURE)?;
    let cases = fixture["Cases"]
        .as_array()
        .ok_or_else(|| std::io::Error::other("IQ3 cases missing"))?;
    for case in cases.iter().filter(|value| value["Scenario"] == 4) {
        let kind = kind(case)?;
        let packed = bytes(&case["WeightsHex"])?;
        let mut weights = tensor(kind, 768, packed.clone());
        weights.dimensions.push(2);
        weights.data = Arc::from([packed.as_slice(), packed.as_slice()].concat());
        let input: Vec<f32> = (0_i16..768)
            .map(|index| f32::from(index % 29 - 14) / 7.0)
            .collect();
        let activations = kernels::quantize_row(15, &input)?;
        let expected = kernels::packed_dot(kind, 768, &packed, &activations)?;
        assert_eq!(kernels::matvec(&weights, &input)?, [expected, expected]);
        let mut scratch = kernels::MatvecScratch::default();
        let mut prepared = scratch.prepare(&weights, &input)?;
        let mut output = Vec::new();
        prepared.matvec_into(&weights, &mut output)?;
        assert_eq!(output, [expected, expected]);
        assert!(kernels::quantize_row(kind, &input[..767]).is_err());
        assert!(kernels::quantize_row_weighted(kind, &input, &[1.0; 767]).is_err());
        assert!(kernels::row(&weights, 2).is_err());
        let size = if kind == 18 { 98 } else { 110 };
        for bytes in [0, size - 1, size + 1] {
            let malformed = tensor(kind, 256, vec![0; bytes]);
            assert!(kernels::validate_tensor(&malformed).is_err());
            let mut output = vec![1.0];
            assert!(kernels::row_into(&malformed, 0, &mut output).is_err());
            assert_eq!(output, Vec::<f32>::new());
        }
        for bits in [0x7c00_u16, 0xfc00, 0x7e01] {
            let mut invalid = vec![0; size];
            invalid[..2].copy_from_slice(&bits.to_le_bytes());
            assert!(kernels::validate_tensor(&tensor(kind, 256, invalid.clone())).is_err());
            assert!(kernels::packed_dot(kind, 256, &invalid, &activations[..292]).is_err());
        }
        let mut invalid_activation = activations[..292].to_vec();
        invalid_activation[..4].copy_from_slice(&f32::NAN.to_le_bytes());
        assert!(kernels::packed_dot(kind, 256, &packed[..size], &invalid_activation).is_err());
        assert!(kernels::packed_dot(kind, 0, &[], &[]).is_err());
        assert!(kernels::packed_dot(kind, 255, &packed[..size], &activations[..292]).is_err());
        assert!(kernels::packed_dot(kind, 256, &packed[..size], &activations[..291]).is_err());
    }
    Ok(())
}

#[test]
fn cpu_graph_uses_the_packed_iq3_activation_contract() -> Result<()> {
    let fixture: Value = serde_json::from_str(FIXTURE)?;
    let cases = fixture["Cases"]
        .as_array()
        .ok_or_else(|| std::io::Error::other("IQ3 cases missing"))?;
    for case in cases.iter().filter(|value| value["Scenario"] == 4) {
        let kind = kind(case)?;
        let packed = bytes(&case["WeightsHex"])?;
        let input: Vec<f32> = (0_i16..768)
            .map(|index| f32::from(index % 29 - 14) / 7.0)
            .collect();
        let expected =
            kernels::packed_dot(kind, 768, &packed, &kernels::quantize_row(15, &input)?)?;
        let mut backend = CpuBackend::new();
        let weights = backend.tensor(1, i32::try_from(kind)?, [768, 1, 1, 1])?;
        backend.allocate(1)?;
        backend.set(weights, 0, &packed)?;
        let activations = backend.tensor(1, 0, [768, 1, 1, 1])?;
        backend.set_values(activations, &input)?;
        let Observation::Tensor(Some(output)) = backend.call(
            "ggml_mul_mat",
            &[
                Argument::Context(1),
                Argument::Tensor(weights),
                Argument::Tensor(activations),
            ],
        )?
        else {
            return Err(std::io::Error::other("missing IQ3 graph output").into());
        };
        assert!(backend.supports(output)?);
        let Observation::Graph(graph) = backend.call("ggml_new_graph", &[Argument::Context(1)])?
        else {
            return Err(std::io::Error::other("missing IQ3 graph").into());
        };
        backend.call(
            "ggml_build_forward_expand",
            &[Argument::Graph(graph), Argument::Tensor(output)],
        )?;
        backend.compute(graph)?;
        assert_eq!(backend.values(output)?[0].to_bits(), expected.to_bits());
    }
    Ok(())
}
