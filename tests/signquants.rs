//! Focused Prism `Q1_0`/`Q2_0` native encoder, decoder and packed-dot evidence.

use bonsai_prism_runtime::{Result, kernels, model::Tensor};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::sync::Arc;

fn bytes(value: &Value) -> Result<Vec<u8>> {
    let text = value
        .as_str()
        .ok_or_else(|| std::io::Error::other("sign-quant fixture hex missing"))?;
    text.as_bytes()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| Ok(u8::from_str_radix(std::str::from_utf8(pair)?, 16)?))
        .collect()
}

fn floats(value: &Value) -> Result<Vec<f32>> {
    Ok(bytes(value)?
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| f32::from_le_bytes(*bytes))
        .collect())
}

fn tensor(kind: u32, count: usize, bytes: Vec<u8>) -> Tensor {
    Tensor {
        name: "sign-quant".to_owned(),
        kind,
        dimensions: vec![count],
        data: Arc::from(bytes),
    }
}

#[test]
fn native_reference_encoders_decoders_dots_and_matvec_agree() -> Result<()> {
    let fixture: Value = serde_json::from_str(include_str!("fixtures/upstream-signquants.json"))?;
    let cases = fixture["Cases"]
        .as_array()
        .ok_or_else(|| std::io::Error::other("sign-quant fixture cases missing"))?;
    assert_eq!(cases.len(), 18);
    for case in cases {
        let kind = u32::try_from(
            case["TypeId"]
                .as_u64()
                .ok_or_else(|| std::io::Error::other("sign-quant fixture kind missing"))?,
        )?;
        let packed = bytes(&case["OutputHex"])?;
        let input = floats(&case["InputHex"])?;
        let weights = tensor(kind, 256, packed.clone());
        kernels::validate_tensor(&weights)?;
        assert_eq!(
            kernels::quantize_row(kind, &input)?,
            packed,
            "reference encoding {}",
            case["Name"]
        );
        assert_eq!(
            kernels::quantize_row_weighted(kind, &input, &[0.25; 256])?,
            packed,
            "native Q1_0/Q2_0 importance is unused"
        );
        assert_eq!(
            kernels::row(&weights, 0)?
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>(),
            bytes(&case["DecodedHex"])?,
            "decode {}",
            case["Name"]
        );
        let activations = bytes(&case["ActivationsHex"])?;
        assert_eq!(
            kernels::packed_dot(kind, 256, &packed, &activations)?
                .to_le_bytes()
                .as_slice(),
            bytes(&case["GenericDotHex"])?,
            "native packed dot {}",
            case["Name"]
        );
        let activation_input = floats(&case["ActivationInputHex"])?;
        assert_eq!(kernels::quantize_row(8, &activation_input)?, activations);
        assert_eq!(
            kernels::matvec(&weights, &activation_input)?[0]
                .to_le_bytes()
                .as_slice(),
            bytes(&case["GenericDotHex"])?,
            "native activation matvec {}",
            case["Name"]
        );
        // Multiple rows and shared preparation retain the same Q8_0 contract.
        let pair = Tensor {
            dimensions: vec![256, 2],
            data: Arc::from([packed.as_slice(), packed.as_slice()].concat()),
            ..weights
        };
        let mut scratch = kernels::MatvecScratch::default();
        let mut prepared = scratch.prepare(&pair, &activation_input)?;
        let mut output = Vec::new();
        prepared.matvec_into(&pair, &mut output)?;
        assert_eq!(output.len(), 2);
        assert_eq!(output[0].to_bits(), output[1].to_bits());
    }
    Ok(())
}

#[test]
fn every_payload_byte_at_every_position_matches_native_decode_and_dot_hashes() -> Result<()> {
    const SCALES: [u16; 8] = [0, 0x8000, 1, 0x8001, 0x3c00, 0xbc00, 0x7bff, 0xfbff];
    let fixture: Value = serde_json::from_str(include_str!("fixtures/upstream-signquants.json"))?;
    let records = fixture["Arbitrary"]
        .as_array()
        .ok_or_else(|| std::io::Error::other("arbitrary sign-quant fixture missing"))?;
    assert_eq!(records.len(), 2);
    for record in records {
        let kind = u32::try_from(
            record["TypeId"]
                .as_u64()
                .ok_or_else(|| std::io::Error::other("arbitrary sign-quant kind missing"))?,
        )?;
        let count = if kind == 41 { 128 } else { 64 };
        let mut decoded_hash = Sha256::new();
        let mut dot_hash = Sha256::new();
        for position in 0..16 {
            for byte in 0..=255_u8 {
                let mut packed = [0_u8; 18];
                packed[..2]
                    .copy_from_slice(&SCALES[(usize::from(byte) + position) % 8].to_le_bytes());
                for (index, value) in packed[2..].iter_mut().enumerate() {
                    *value = u8::try_from((index * 37 + position * 17) % 256)?;
                }
                packed[2 + position] = byte;
                let weights = tensor(kind, count, packed.to_vec());
                kernels::validate_tensor(&weights)?;
                for value in kernels::row(&weights, 0)? {
                    decoded_hash.update(value.to_le_bytes());
                }
                let mut activation = Vec::new();
                for group in 0..count / 32 {
                    activation.extend(SCALES[(position + group + 4) % 8].to_le_bytes());
                    for index in 0..32 {
                        activation.push(u8::try_from(
                            (usize::from(byte) + index * 13 + group * 23) % 256,
                        )?);
                    }
                }
                dot_hash
                    .update(kernels::packed_dot(kind, count, &packed, &activation)?.to_le_bytes());
            }
        }
        assert_eq!(record["Blocks"], 4096);
        assert_eq!(
            decoded_hash.finalize().as_slice(),
            bytes(&record["DecodedSha256"])?
        );
        assert_eq!(dot_hash.finalize().as_slice(), bytes(&record["DotSha256"])?);
    }
    Ok(())
}

#[test]
fn malformed_rows_scales_activations_and_quantization_fail_cleanly() {
    for (kind, count) in [(41, 128), (42, 64)] {
        for size in [0, 17, 19, 34] {
            let malformed = tensor(kind, count, vec![0; size]);
            assert!(kernels::validate_tensor(&malformed).is_err());
            let mut output = vec![1.0];
            assert!(kernels::row_into(&malformed, 0, &mut output).is_err());
            assert_eq!(output, Vec::<f32>::new());
        }
        let mut packed = [0_u8; 18];
        packed[..2].copy_from_slice(&0x3c00_u16.to_le_bytes());
        let mut weights = tensor(kind, count, packed.to_vec());
        assert!(kernels::row(&weights, 1).is_err());
        weights.dimensions[0] = count - 1;
        assert!(kernels::validate_tensor(&weights).is_err());
        let activations = vec![0; count / 32 * 34];
        assert!(kernels::packed_dot(kind, 0, &[], &[]).is_err());
        assert!(kernels::packed_dot(kind, count - 1, &packed, &activations).is_err());
        assert!(kernels::packed_dot(kind, count, &packed, &activations[..1]).is_err());
        for bits in [0x7c00_u16, 0xfc00, 0x7e01] {
            packed[..2].copy_from_slice(&bits.to_le_bytes());
            assert!(kernels::validate_tensor(&tensor(kind, count, packed.to_vec())).is_err());
            assert!(kernels::packed_dot(kind, count, &packed, &activations).is_err());
        }
        packed[..2].copy_from_slice(&0x3c00_u16.to_le_bytes());
        let mut invalid_activation = activations;
        invalid_activation[..2].copy_from_slice(&0x7c00_u16.to_le_bytes());
        assert!(kernels::packed_dot(kind, count, &packed, &invalid_activation).is_err());
        assert!(kernels::quantize_row(kind, &[]).is_err());
        assert!(kernels::quantize_row(kind, &vec![0.0; count - 1]).is_err());
        for value in [f32::NAN, f32::INFINITY, f32::MAX, 65_536.0] {
            assert!(kernels::quantize_row(kind, &vec![value; count]).is_err());
        }
        let input = vec![0.0; count];
        assert!(kernels::quantize_row_weighted(kind, &input, &[0.0]).is_err());
        assert!(kernels::quantize_row_weighted(kind, &input, &vec![-1.0; count]).is_err());
        assert!(kernels::quantize_row_weighted(kind, &input, &vec![f32::NAN; count]).is_err());
    }
}
