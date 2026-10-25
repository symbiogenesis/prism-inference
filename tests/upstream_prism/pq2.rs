//! Original PQ2 shape cases with an independent Rust reference calculation.

use serde_json::Value;

use super::engine::{Engine, Output, Pq2Input, Request};
use super::fixture::Fixture;
use super::{PQ2_ID, Result, invalid, result};

const INPUTS: &[u8] = include_bytes!("pq2-inputs.bin");

fn inputs(bytes: &[u8]) -> Result<Vec<Request>> {
    let mut fixture = Fixture::new(bytes, *b"BSPQ2I1\n")?;
    if fixture.u32()? != 15 {
        return Err(invalid("PQ2 inputs must contain all 15 upstream shapes"));
    }
    let mut requests = Vec::with_capacity(15);
    for k in [128_usize, 256, 384, 512, 640] {
        for n in [1_usize, 4, 16] {
            let m = 4;
            let shape = [fixture.u32()?, fixture.u32()?, fixture.u32()?];
            let packed_size = k / 128 * m * 34;
            if shape != [u32::try_from(k)?, u32::try_from(m)?, u32::try_from(n)?]
                || fixture.u32()? != u32::try_from(packed_size)?
            {
                return Err(invalid("PQ2 input shape/order/packed extent mismatch"));
            }
            requests.push(Request::Pq2(Pq2Input {
                k,
                m,
                n,
                packed: fixture.block(packed_size)?,
                activations: fixture.floats(k * n)?,
            }));
        }
    }
    fixture.finish()?;
    Ok(requests)
}

// Deliberately independent of bonsai_ternary and runtime row decoders: binary16
// arithmetic followed by the format's little-endian two-bit symbol traversal.
fn decode(packed: &[u8]) -> Result<Vec<f32>> {
    let (blocks, remainder) = packed.as_chunks::<34>();
    if !remainder.is_empty() {
        return Err(invalid("incomplete PQ2 reference block"));
    }
    let mut decoded = Vec::with_capacity(blocks.len() * 128);
    for block in blocks {
        let half = u16::from_le_bytes([block[0], block[1]]);
        let exponent = i32::from((half >> 10) & 31);
        if exponent == 31 {
            return Err(invalid("nonfinite PQ2 reference scale"));
        }
        let fraction = f32::from(half & 1023);
        let magnitude = if exponent == 0 {
            fraction * 2.0_f32.powi(-24)
        } else {
            (1.0 + fraction / 1024.0) * 2.0_f32.powi(exponent - 15)
        };
        let scale = if half & 0x8000 == 0 {
            magnitude
        } else {
            -magnitude
        };
        for index in 0..128 {
            let symbol = (block[2 + index / 4] >> (2 * (index % 4))) & 3;
            if symbol == 3 {
                return Err(invalid("reserved symbol in PQ2 input corpus"));
            }
            decoded.push(scale * f32::from(i16::from(symbol) - 1));
        }
    }
    Ok(decoded)
}

#[allow(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::suboptimal_flops,
    reason = "Preserve upstream double accumulation and its final rounding to float."
)]
fn reference(input: &Pq2Input) -> Result<Vec<f32>> {
    let decoded = decode(&input.packed)?;
    let mut output = Vec::with_capacity(input.m * input.n);
    for j in 0..input.n {
        for i in 0..input.m {
            let mut sum = 0.0_f64;
            for column in 0..input.k {
                sum += f64::from(decoded[i * input.k + column])
                    * f64::from(input.activations[j * input.k + column]);
            }
            output.push(sum as f32);
        }
    }
    Ok(output)
}

// Preserve upstream rel_rms, including its all-zero reference behavior.
#[allow(
    clippy::suboptimal_flops,
    reason = "Preserve upstream separate multiplication and addition."
)]
fn rel_rms(actual: &[f32], reference: &[f32]) -> f64 {
    let mut squared_error = 0.0;
    let mut squared_reference = 0.0;
    for (&actual, &reference) in actual.iter().zip(reference) {
        let delta = f64::from(actual) - f64::from(reference);
        squared_error += delta * delta;
        squared_reference += f64::from(reference) * f64::from(reference);
    }
    if squared_reference == 0.0 {
        if squared_error == 0.0 {
            0.0
        } else {
            f64::INFINITY
        }
    } else {
        (squared_error / squared_reference).sqrt()
    }
}

pub(super) fn run(engine: &Engine) -> Result<Value> {
    let requests = inputs(INPUTS)?;
    let actual = engine.execute("pq2-row-shapes", &requests)?;
    if actual.len() != requests.len() {
        return Err(invalid("PQ2 engine returned the wrong shape count"));
    }
    let mut failures = Vec::new();
    let mut worst = 0.0_f64;
    for (request, actual) in requests.iter().zip(actual) {
        let Request::Pq2(input) = request else {
            return Err(invalid("PQ2 operation type mismatch"));
        };
        let Output::Pq2(actual) = actual else {
            return Err(invalid("PQ2 engine result type mismatch"));
        };
        let label = format!("K={} M={} N={}", input.k, input.m, input.n);
        let reference = reference(input)?;
        match actual {
            Ok(actual) if actual.len() == reference.len() => {
                let error = rel_rms(&actual, &reference);
                worst = worst.max(error);
                if !error.is_finite() || error >= 0.05 {
                    failures.push(format!("{label}: rel_rms={error:.8}"));
                }
            }
            Ok(_) => failures.push(format!("{label}: output extent mismatch")),
            Err(error) => failures.push(format!("{label}: {error}")),
        }
    }
    Ok(result(
        PQ2_ID,
        failures.is_empty(),
        format!(
            "15 upstream shapes; K=128/256/384/512/640, M=4, N=1/4/16; \
             rel_rms < 0.05; worst={worst:.8}; failed_shapes={}; {}",
            failures.len(),
            failures.join("; ")
        ),
    ))
}

#[cfg(test)]
mod tests {
    use sha2::{Digest, Sha256};

    use super::*;

    #[test]
    fn frozen_corpus_contains_only_original_inputs_in_original_order() -> Result<()> {
        assert_eq!(INPUTS.len(), 167_652);
        assert_eq!(
            Sha256::digest(INPUTS).as_slice(),
            &[
                0xa3, 0x7e, 0x4e, 0x61, 0x33, 0x36, 0x47, 0xff, 0x52, 0x8a, 0xc5, 0x97, 0xe8, 0xbb,
                0xa7, 0x5b, 0x94, 0x5a, 0xc0, 0x25, 0x54, 0x30, 0xa8, 0x73, 0x32, 0x71, 0x9f, 0x34,
                0xc5, 0x6b, 0x5e, 0xf1,
            ]
        );
        assert_eq!(inputs(INPUTS)?.len(), 15);
        assert!(inputs(&INPUTS[..INPUTS.len() - 1]).is_err());
        let mut corrupted = INPUTS.to_vec();
        corrupted[12..16].copy_from_slice(&256_u32.to_le_bytes());
        assert!(inputs(&corrupted).is_err());
        Ok(())
    }

    #[test]
    fn reference_decodes_symbol_and_scale_without_runtime_helpers() -> Result<()> {
        let mut block = [0x99; 34];
        block[..2].copy_from_slice(&0x3800_u16.to_le_bytes());
        assert_eq!(&decode(&block)?[..4], &[0.0, 0.5, 0.0, 0.5]);
        assert_eq!(rel_rms(&[0.0], &[0.0]), 0.0);
        assert!(rel_rms(&[1.0], &[0.0]).is_infinite());
        assert!(decode(&block[..33]).is_err());
        block[2] = 3;
        assert!(decode(&block).is_err());
        Ok(())
    }
}
