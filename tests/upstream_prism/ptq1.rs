//! Original 20,000 LCG blocks and CPU traversal, owned by safe Rust.

use serde_json::Value;

use super::engine::{Engine, Output, Request};
use super::{PTQ1_ID, Result, invalid, result};

const TRIALS: usize = 20_000;

fn inputs() -> Vec<[u8; 28]> {
    let mut seed = 12_345_u32;
    (0..TRIALS)
        .map(|_| {
            let mut block = [0; 28];
            for byte in &mut block[..26] {
                seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                *byte = seed.to_le_bytes()[2];
            }
            // Upstream's integer test never reads d. A unit scale permits the
            // unchanged exact trit assertions against production float output.
            block[26..].copy_from_slice(&0x3c00_u16.to_le_bytes());
            block
        })
        .collect()
}

// Preserve cpu_ref's stage traversal, rather than implementing the accessor's
// index arithmetic or calling a runtime codec which is itself under test.
fn cpu_reference(block: &[u8; 28]) -> Vec<f32> {
    let powers = [1_u8, 3, 9, 27, 81];
    let mut output = Vec::with_capacity(128);
    let mut j = 0;
    for width in [32, 16, 8] {
        while j + width <= 24 {
            for power in powers {
                for m in 0..width {
                    let q = block[j + m].wrapping_mul(power);
                    output.push(f32::from((u16::from(q) * 3) >> 8) - 1.0);
                }
            }
            j += width;
        }
    }
    for power in &powers[..4] {
        for h in 0..2 {
            let q = block[24 + h].wrapping_mul(*power);
            output.push(f32::from((u16::from(q) * 3) >> 8) - 1.0);
        }
    }
    output
}

// Independent transcription of the upstream CUDA device accessor's integer
// indexing. The original host test compared this with cpu_ref without nvcc.
fn element_reference(block: &[u8; 28], element: usize) -> f32 {
    let (byte, digit) = if element < 80 {
        (block[element & 15], element >> 4)
    } else if element < 120 {
        let position = element - 80;
        (block[16 + (position & 7)], position >> 3)
    } else {
        let position = element - 120;
        (block[24 + (position & 1)], position >> 1)
    };
    let mut value = u32::from(byte);
    for index in 0..4 {
        if index < digit {
            value = (value * 3) & 0xff;
        }
    }
    // The product's high byte is one of 0/1/2, represented exactly as f32.
    f32::from(((value * 3) >> 8).to_le_bytes()[0]) - 1.0
}

pub(super) fn run(engine: &Engine) -> Result<Value> {
    let blocks = inputs();
    let requests = [Request::Ptq1(blocks)];
    let mut outputs = engine.execute("ptq1-element-map", &requests)?;
    if outputs.len() != 1 {
        return Err(invalid("PTQ1 engine returned the wrong operation count"));
    }
    let Some(Output::Ptq1(actual)) = outputs.pop() else {
        return Err(invalid("PTQ1 engine result type mismatch"));
    };
    let Request::Ptq1(blocks) = &requests[0] else {
        return Err(invalid("PTQ1 input type mismatch"));
    };
    if actual.len() != TRIALS {
        return Err(invalid("PTQ1 engine returned the wrong block count"));
    }
    let mut rejected = 0;
    let mut compared = 0;
    let mut mismatches = 0;
    let mut mapping_mismatches = 0;
    let mut first_failure = None;
    for (trial, (block, actual)) in blocks.iter().zip(actual).enumerate() {
        let expected = cpu_reference(block);
        for (element, &reference) in expected.iter().enumerate() {
            let accessor = element_reference(block, element);
            if accessor.to_bits() != reference.to_bits() {
                mapping_mismatches += 1;
                first_failure.get_or_insert_with(|| {
                    format!("trial={trial} element={element}: accessor={accessor} cpu={reference}")
                });
            }
        }
        match actual {
            Ok(actual) => {
                if actual.len() != 128 {
                    return Err(invalid("PTQ1 row decoder returned invalid extent"));
                }
                for (element, (&actual, &reference)) in actual.iter().zip(&expected).enumerate() {
                    compared += 1;
                    // Unit scale makes every original integer equality exact.
                    if actual.to_bits() != reference.to_bits() {
                        mismatches += 1;
                        first_failure.get_or_insert_with(|| {
                            format!("trial={trial} element={element}: {actual} != {reference}")
                        });
                    }
                }
            }
            Err(error) => {
                rejected += 1;
                first_failure.get_or_insert_with(|| format!("trial={trial}: {error}"));
            }
        }
    }
    Ok(result(
        PTQ1_ID,
        mismatches == 0 && rejected == 0 && mapping_mismatches == 0,
        format!(
            "20000 original LCG blocks; 2560000 required positions; compared={compared}; \
             mismatches={mismatches}; mapping_mismatches={mapping_mismatches}; \
             rejected_blocks={rejected}; {}",
            first_failure.unwrap_or_default()
        ),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn original_lcg_seed_wrapping_and_stage_extents_are_preserved() {
        let blocks = inputs();
        assert_eq!(blocks.len(), TRIALS);
        assert_eq!(&blocks[0][..8], &[220, 4, 101, 170, 31, 173, 29, 90]);
        assert!(blocks.iter().all(|block| cpu_reference(block).len() == 128));
        assert_eq!(cpu_reference(&[0; 28]), vec![-1.0; 128]);
    }
}
