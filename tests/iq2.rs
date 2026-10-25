//! Focused IQ2 codebook decoder and native integer-dot observations.

use bonsai_prism_runtime::{Result, kernels, model::Tensor};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::sync::Arc;

fn bytes(value: &Value) -> Result<Vec<u8>> {
    let text = value
        .as_str()
        .ok_or_else(|| std::io::Error::other("IQ2 fixture hex missing"))?;
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
        name: "IQ2".to_owned(),
        kind,
        dimensions: vec![count],
        data: Arc::from(bytes),
    }
}

fn type_id(record: &Value) -> Result<u32> {
    Ok(u32::try_from(record["TypeId"].as_u64().ok_or_else(
        || std::io::Error::other("IQ2 fixture kind missing"),
    )?)?)
}

#[test]
fn native_iq2_decoders_packed_dots_and_prepared_matvec_preserve_bytes() -> Result<()> {
    let fixture: Value = serde_json::from_str(include_str!("fixtures/upstream-iq2.json"))?;
    let cases = fixture["Cases"]
        .as_array()
        .ok_or_else(|| std::io::Error::other("IQ2 fixture cases missing"))?;
    assert_eq!(cases.len(), 24);
    for case in cases {
        let kind = type_id(case)?;
        let packed = bytes(&case["OutputHex"])?;
        let weights = tensor(kind, 512, packed.clone());
        kernels::validate_tensor(&weights)?;
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
            kernels::packed_dot(kind, 512, &packed, &activations)?
                .to_le_bytes()
                .as_slice(),
            bytes(&case["GenericDotHex"])?,
            "native packed dot {}",
            case["Name"]
        );
        let activation_input = floats(&case["ActivationInputHex"])?;
        assert_eq!(kernels::quantize_row(15, &activation_input)?, activations);
        assert_eq!(
            kernels::matvec(&weights, &activation_input)?[0]
                .to_le_bytes()
                .as_slice(),
            bytes(&case["GenericDotHex"])?,
            "native activation matvec {}",
            case["Name"]
        );
        let pair = Tensor {
            dimensions: vec![512, 2],
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
fn every_iq2_payload_byte_matches_native_decoder_and_integer_dot_hashes() -> Result<()> {
    const SCALES: [u16; 8] = [0, 0x8000, 1, 0x8001, 0x3c00, 0xbc00, 0x7bff, 0xfbff];
    const ACTIVATION_SCALES: [f32; 8] = [
        0.0,
        -0.0,
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
        1.0,
        -1.0,
        123.125,
        -123.125,
    ];
    let fixture: Value = serde_json::from_str(include_str!("fixtures/upstream-iq2.json"))?;
    let records = fixture["Arbitrary"]
        .as_array()
        .ok_or_else(|| std::io::Error::other("arbitrary IQ2 fixture missing"))?;
    assert_eq!(records.len(), 3);
    for record in records {
        let kind = type_id(record)?;
        let size = match kind {
            16 => 66,
            17 => 74,
            _ => 82,
        };
        assert_eq!(record["BlockBytes"], size);
        let mut decoded_hash = Sha256::new();
        let mut dot_hash = Sha256::new();
        for position in 0..size - 2 {
            for byte in 0..=255_u8 {
                let mut packed = vec![0_u8; size];
                packed[..2]
                    .copy_from_slice(&SCALES[(usize::from(byte) + position) % 8].to_le_bytes());
                for (index, value) in packed.iter_mut().enumerate().skip(2) {
                    *value = u8::try_from((index * 37 + position * 17) % 256)?;
                }
                packed[2 + position] = byte;
                let weights = tensor(kind, 256, packed.clone());
                kernels::validate_tensor(&weights)?;
                let decoded = kernels::row(&weights, 0)?
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect::<Vec<_>>();
                decoded_hash.update(decoded);
                let mut activation = Vec::with_capacity(292);
                activation.extend(ACTIVATION_SCALES[(position + 4) % 8].to_le_bytes());
                for index in 0..256 {
                    activation.push(u8::try_from(
                        (usize::from(byte) + index * 13 + position * 23) % 256,
                    )?);
                }
                // These fields are deliberately unrelated to the values: IQ2's
                // native dot consumes the packed values, not Q8_K group sums.
                for index in 0..16 {
                    activation.extend(i16::try_from(index * 911 + position)?.to_le_bytes());
                }
                dot_hash
                    .update(kernels::packed_dot(kind, 256, &packed, &activation)?.to_le_bytes());
            }
        }
        assert_eq!(record["Blocks"], (size - 2) * 256);
        assert_eq!(
            decoded_hash.finalize().as_slice(),
            bytes(&record["DecodedSha256"])?
        );
        assert_eq!(dot_hash.finalize().as_slice(), bytes(&record["DotSha256"])?);
    }
    Ok(())
}

#[test]
fn iq2_rejects_malformed_extents_nonfinite_scales_and_unavailable_encoders() -> Result<()> {
    for (kind, size) in [(16, 66), (17, 74), (22, 82)] {
        for length in [0, size - 1, size + 1, size * 2 - 1] {
            let malformed = tensor(kind, 256, vec![0; length]);
            assert!(kernels::validate_tensor(&malformed).is_err());
            let mut output = vec![1.0];
            assert!(kernels::row_into(&malformed, 0, &mut output).is_err());
            assert_eq!(output, Vec::<f32>::new());
        }
        let mut packed = vec![0_u8; size];
        packed[..2].copy_from_slice(&0x3c00_u16.to_le_bytes());
        let weights = tensor(kind, 256, packed.clone());
        assert!(kernels::row(&weights, 1).is_err());
        let activations = [0_u8; 292];
        assert!(kernels::packed_dot(kind, 0, &[], &[]).is_err());
        assert!(kernels::packed_dot(kind, 255, &packed, &activations).is_err());
        assert!(kernels::packed_dot(kind, 256, &packed[..size - 1], &activations).is_err());
        assert!(kernels::packed_dot(kind, 256, &packed, &activations[..291]).is_err());
        for bits in [0x7c00_u16, 0xfc00, 0x7e01] {
            packed[..2].copy_from_slice(&bits.to_le_bytes());
            assert!(kernels::validate_tensor(&tensor(kind, 256, packed.clone())).is_err());
            assert!(kernels::packed_dot(kind, 256, &packed, &activations).is_err());
            let pair = Tensor {
                dimensions: vec![256, 2],
                data: Arc::from([weights.data.as_ref(), packed.as_slice()].concat()),
                ..weights.clone()
            };
            let mut output = vec![1.0];
            assert!(
                kernels::matvec_into(
                    &pair,
                    &[1.0; 256],
                    &mut output,
                    &mut kernels::MatvecScratch::default()
                )
                .is_err()
            );
            assert_eq!(output, Vec::<f32>::new());
        }
        packed[..2].copy_from_slice(&0x3c00_u16.to_le_bytes());
        let mut invalid_activation = activations;
        invalid_activation[..4].copy_from_slice(&f32::NAN.to_le_bytes());
        assert!(kernels::packed_dot(kind, 256, &packed, &invalid_activation).is_err());
        if kind == 22 {
            assert_eq!(kernels::quantize_row(kind, &[0.0; 256])?, vec![0; size]);
            assert_eq!(
                kernels::quantize_row_weighted(kind, &[0.0; 256], &[1.0; 256])?,
                vec![0; size]
            );
        } else {
            assert!(kernels::quantize_row(kind, &[0.0; 256]).is_err());
            assert!(kernels::quantize_row_weighted(kind, &[0.0; 256], &[1.0; 256]).is_err());
        }
    }
    // A shared preparation can serve each IQ2 codebook without changing its
    // native Q8_K quantizer or allocating a decoded tensor.
    let mut scratch = kernels::MatvecScratch::default();
    let first = tensor(16, 256, vec![0; 66]);
    let prepared = scratch.prepare(&first, &[1.0; 256])?;
    assert!(prepared.is_compatible(&tensor(17, 256, vec![0; 74]))?);
    assert!(prepared.is_compatible(&tensor(22, 256, vec![0; 82]))?);
    Ok(())
}
