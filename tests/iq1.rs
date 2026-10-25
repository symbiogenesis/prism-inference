//! Native `IQ1_S`/`IQ1_M` table, scale, arbitrary-byte and `Q8_K` dot regressions.

use bonsai_prism_runtime::{Result, kernels, model::Tensor};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::sync::Arc;

const SCALES: [u16; 8] = [0, 0x8000, 1, 0x8001, 0x3c00, 0xbc00, 0x7bff, 0xfbff];

fn bytes(value: &Value) -> Result<Vec<u8>> {
    value
        .as_str()
        .ok_or_else(|| std::io::Error::other("IQ1 fixture hex missing"))?
        .as_bytes()
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
        .map(|bits| f32::from_le_bytes(*bits))
        .collect())
}

fn tensor(kind: u32, count: usize, bytes: Vec<u8>) -> Tensor {
    Tensor {
        name: "iq1-native".to_owned(),
        kind,
        dimensions: vec![count],
        data: Arc::from(bytes),
    }
}

fn set_scale(kind: u32, block: &mut [u8], bits: u16) {
    if kind == 19 {
        block[..2].copy_from_slice(&bits.to_le_bytes());
    } else {
        for index in 0..4 {
            let offset = 48 + 2 * index;
            let value = u16::from_le_bytes([block[offset], block[offset + 1]]);
            let value = (value & 0x0fff) | (((bits >> (4 * index)) & 15) << 12);
            block[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
        }
    }
}

fn activation(seed: usize) -> Result<Vec<u8>> {
    let mut bytes = (-0.125_f32).to_le_bytes().to_vec();
    for index in 0..256 {
        bytes.push(u8::try_from((seed + index * 13) % 256)?);
    }
    for index in 0..16 {
        let value = i64::try_from(seed * 29 + index * 97)? - 17_000;
        bytes.extend(u16::try_from(value.rem_euclid(65_536))?.to_le_bytes());
    }
    Ok(bytes)
}

#[test]
fn native_iq1_decoding_packed_dots_and_q8k_matvec_match() -> Result<()> {
    let fixture: Value = serde_json::from_str(include_str!("fixtures/upstream-iq1.json"))?;
    let cases = fixture["Cases"]
        .as_array()
        .ok_or_else(|| std::io::Error::other("IQ1 fixture cases missing"))?;
    assert_eq!(cases.len(), 16);
    for case in cases {
        let kind = u32::try_from(
            case["TypeId"]
                .as_u64()
                .ok_or_else(|| std::io::Error::other("IQ1 type missing"))?,
        )?;
        let count = usize::try_from(
            case["Count"]
                .as_u64()
                .ok_or_else(|| std::io::Error::other("IQ1 count missing"))?,
        )?;
        let packed = bytes(&case["WeightsHex"])?;
        let weights = tensor(kind, count, packed.clone());
        kernels::validate_tensor(&weights)?;
        assert_eq!(
            kernels::row(&weights, 0)?
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>(),
            bytes(&case["DecodedHex"])?
        );
        let activations = bytes(&case["ActivationsHex"])?;
        let expected = bytes(&case["GenericDotHex"])?;
        assert_eq!(
            kernels::packed_dot(kind, count, &packed, &activations)?
                .to_le_bytes()
                .as_slice(),
            expected,
            "{}",
            case["Name"]
        );
        if case["ActivationInputHex"].is_string() {
            let input = floats(&case["ActivationInputHex"])?;
            assert_eq!(kernels::quantize_row(15, &input)?, activations);
            assert_eq!(
                kernels::matvec(&weights, &input)?[0]
                    .to_le_bytes()
                    .as_slice(),
                expected
            );
            let pair = Tensor {
                dimensions: vec![count, 2],
                data: Arc::from([packed.as_slice(), packed.as_slice()].concat()),
                ..weights
            };
            let mut scratch = kernels::MatvecScratch::default();
            let mut prepared = scratch.prepare(&pair, &input)?;
            let mut output = Vec::new();
            prepared.matvec_into(&pair, &mut output)?;
            assert_eq!(output.len(), 2);
            assert_eq!(output[0].to_bits(), output[1].to_bits());
        }
    }
    Ok(())
}

#[test]
fn every_iq1_grid_entry_and_arbitrary_payload_byte_match_native_hashes() -> Result<()> {
    check_native_hashes(&[0, 1])
}

#[test]
fn all_finite_iq1_scale_patterns_match_native_and_nonfinite_scales_fail() -> Result<()> {
    check_native_hashes(&[2])?;
    for kind in [19, 29] {
        let size = if kind == 19 { 50 } else { 56 };
        for bits in 0_u16..=u16::MAX {
            if bits & 0x7c00 != 0x7c00 {
                continue;
            }
            let mut block = vec![0; size];
            set_scale(kind, &mut block, bits);
            let weights = tensor(kind, 256, block);
            assert!(kernels::validate_tensor(&weights).is_err());
            assert!(kernels::row(&weights, 0).is_err());
        }
    }
    Ok(())
}

fn check_native_hashes(modes: &[u64]) -> Result<()> {
    let fixture: Value = serde_json::from_str(include_str!("fixtures/upstream-iq1.json"))?;
    let records = fixture["Arbitrary"]
        .as_array()
        .ok_or_else(|| std::io::Error::other("IQ1 hash records missing"))?;
    assert_eq!(records.len(), 6);
    for record in records {
        let mode = record["Mode"]
            .as_u64()
            .ok_or_else(|| std::io::Error::other("IQ1 mode missing"))?;
        if !modes.contains(&mode) {
            continue;
        }
        let kind = u32::try_from(
            record["TypeId"]
                .as_u64()
                .ok_or_else(|| std::io::Error::other("IQ1 hash kind missing"))?,
        )?;
        let size = if kind == 19 { 50 } else { 56 };
        let iterations = match mode {
            0 => (if kind == 19 { 48 } else { 56 }) * 256,
            1 => 2048,
            _ => 65_536,
        };
        let mut decoded_hash = Sha256::new();
        let mut dot_hash = Sha256::new();
        let mut blocks = 0;
        for iteration in 0..iterations {
            let mut block = (0..size)
                .map(|index| u8::try_from((index * 37 + (iteration / 256) * 17) % 256))
                .collect::<std::result::Result<Vec<_>, _>>()?;
            if mode == 0 {
                let position = iteration / 256 + if kind == 19 { 2 } else { 0 };
                block[position] = u8::try_from(iteration % 256)?;
                set_scale(
                    kind,
                    &mut block,
                    SCALES[(iteration % 256 + iteration / 256) % 8],
                );
            } else if mode == 1 {
                for group in 0..8 {
                    for part in 0..4 {
                        block[usize::from(kind == 19) * 2 + 4 * group + part] =
                            u8::try_from(iteration % 256)?;
                    }
                    if kind == 19 {
                        let high = u16::try_from(
                            (iteration >> 8) * 585 + ((group % 8) << 12) + ((group % 2) << 15),
                        )?;
                        block[34 + 2 * group..36 + 2 * group].copy_from_slice(&high.to_le_bytes());
                    } else {
                        let high = u8::try_from(
                            (iteration >> 8) * 17 + if group % 2 == 1 { 0x88 } else { 0 },
                        )?;
                        block[32 + 2 * group] = high;
                        block[33 + 2 * group] = high;
                    }
                }
                set_scale(kind, &mut block, 0x3c00);
            } else {
                let bits = u16::try_from(iteration)?;
                if bits & 0x7c00 == 0x7c00 {
                    continue;
                }
                set_scale(kind, &mut block, bits);
            }
            let weights = tensor(kind, 256, block.clone());
            for value in kernels::row(&weights, 0)? {
                decoded_hash.update(value.to_le_bytes());
            }
            dot_hash.update(
                kernels::packed_dot(kind, 256, &block, &activation(iteration)?)?.to_le_bytes(),
            );
            blocks += 1;
        }
        assert_eq!(record["Blocks"], blocks);
        assert_eq!(
            decoded_hash.finalize().as_slice(),
            bytes(&record["DecodedSha256"])?
        );
        assert_eq!(dot_hash.finalize().as_slice(), bytes(&record["DotSha256"])?);
    }
    Ok(())
}

#[test]
fn iq1_malformed_extents_and_unavailable_encoders_return_errors() -> Result<()> {
    for kind in [19, 29] {
        let size = if kind == 19 { 50 } else { 56 };
        for count in [0, size - 1, size + 1, 2 * size - 1] {
            let weights = tensor(kind, 256, vec![0; count]);
            assert!(kernels::validate_tensor(&weights).is_err());
            let mut output = vec![1.0];
            assert!(kernels::row_into(&weights, 0, &mut output).is_err());
            assert_eq!(output, Vec::<f32>::new());
        }
        let weights = tensor(kind, 256, vec![0; size]);
        assert!(kernels::row(&weights, 1).is_err());
        assert!(kernels::packed_dot(kind, 0, &[], &[]).is_err());
        assert!(kernels::packed_dot(kind, 255, &weights.data, &activation(0)?).is_err());
        assert!(kernels::packed_dot(kind, 256, &weights.data, &[0; 291]).is_err());
        let mut invalid = activation(0)?;
        invalid[..4].copy_from_slice(&f32::INFINITY.to_le_bytes());
        assert!(kernels::packed_dot(kind, 256, &weights.data, &invalid).is_err());
        assert!(kernels::quantize_row(kind, &[1.0; 256]).is_err());
        assert!(kernels::quantize_row_weighted(kind, &[1.0; 256], &[1.0; 256]).is_err());
    }
    Ok(())
}
