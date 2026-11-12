//! Independent wire, transform and approximation controls for the scalar codec.

#![expect(
    clippy::suboptimal_flops,
    reason = "Independent numerical oracles deliberately use separate products and additions instead of changing the specified codec arithmetic to fused operations."
)]

use super::{Bits, HEAD_DIM, Quantizer, scalar, tables};
use crate::Result;

fn input_row() -> [f32; HEAD_DIM] {
    let mut row = [0.0; HEAD_DIM];
    for (index, value) in (0_u16..256).zip(&mut row) {
        let linear = f32::from(index) - 127.5;
        *value = linear / 32.0 + (linear * 0.17).sin();
    }
    row
}

fn record(bits: Bits, scale: f32) -> Vec<u8> {
    let quantizer = Quantizer::scalar(bits, 91);
    let mut packed = vec![0_u8; quantizer.encoded_len()];
    packed[..4].copy_from_slice(&scale.to_le_bytes());
    // Build each code one bit at a time, independently of write_code/read_code.
    for index in 0..HEAD_DIM {
        let code = index % usize::from(bits.levels());
        for bit in 0..usize::from(bits.bit_width()) {
            let stream_bit = index * usize::from(bits.bit_width()) + bit;
            if code & (1 << bit) != 0 {
                packed[4 + stream_bit / 8] |= 1 << (stream_bit % 8);
            }
        }
    }
    packed
}

fn near(actual: f32, expected: f32, tolerance: f32) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "actual={actual}, expected={expected}, tolerance={tolerance}"
    );
}

#[test]
fn wire_sizes_bits_and_identity_are_explicit() {
    let three = Quantizer::scalar(Bits::Three, 7);
    let four = Quantizer::scalar(Bits::Four, 7);
    assert_eq!(three.encoded_len(), 100);
    assert_eq!(four.encoded_len(), 132);
    assert_eq!(Bits::Three.bit_width(), 3);
    assert_eq!(Bits::Four.bit_width(), 4);
    assert_ne!(three.identity(), four.identity());
    assert_ne!(
        three.identity(),
        Quantizer::scalar(Bits::Three, 8).identity()
    );
    assert_eq!(three.identity(), Quantizer::new(Bits::Three, 7).identity());
    assert!(!three.uses_neon());
}

#[test]
fn splitmix_stream_matches_independent_known_words() {
    let mut seed = 0;
    assert_eq!(scalar::splitmix64(&mut seed), 0xe220_a839_7b1d_cdaf);
    assert_eq!(scalar::splitmix64(&mut seed), 0x6e78_9e6a_a1b9_65f4);
    assert_eq!(scalar::splitmix64(&mut seed), 0x06c4_5d18_8009_454f);
}

#[test]
fn packing_matches_fixed_little_endian_goldens() {
    let mut three = [0_u8; 3];
    for code in 0_u8..8 {
        scalar::write_code(&mut three, 3, usize::from(code), code);
    }
    assert_eq!(three, [0x88, 0xc6, 0xfa]);
    for code in 0..8 {
        assert_eq!(scalar::read_code(&three, 3, code), code);
    }
    let mut four = [0_u8; 8];
    for code in 0_u8..16 {
        scalar::write_code(&mut four, 4, usize::from(code), code);
    }
    assert_eq!(four, [0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe]);
    for code in 0..16 {
        assert_eq!(scalar::read_code(&four, 4, code), code);
    }
}

#[test]
fn transform_matches_an_independent_dense_signed_hadamard() -> Result<()> {
    let quantizer = Quantizer::scalar(Bits::Four, 42);
    let input = input_row();
    let mut actual = [0.0; HEAD_DIM];
    quantizer.rotate(&input, &mut actual)?;
    let mut expected = input;
    for signs in &quantizer.signs {
        let mut next = [0.0; HEAD_DIM];
        for (index, value) in next.iter_mut().enumerate() {
            let mut sum = 0.0_f64;
            for (channel, (&input, &sign)) in expected.iter().zip(signs).enumerate() {
                let parity = if (index & channel).count_ones() % 2 == 0 {
                    1.0
                } else {
                    -1.0
                };
                sum += f64::from(input) * f64::from(sign) * parity;
            }
            *value = scalar::checked_f32(sum / 16.0)?;
        }
        expected = next;
    }
    for (&actual, &expected) in actual.iter().zip(&expected) {
        near(actual, expected, 0.000_004);
    }
    quantizer.inverse(&mut actual)?;
    for (&actual, &expected) in actual.iter().zip(&input) {
        near(actual, expected, 0.000_004);
    }
    Ok(())
}

#[test]
fn codebook_centroids_satisfy_gaussian_conditional_means() {
    for (table, levels) in [(&tables::THREE, 8), (&tables::FOUR, 16)] {
        for index in 0..levels {
            let left = if index == 0 {
                -10.0
            } else {
                f64::midpoint(f64::from(table[index - 1]), f64::from(table[index]))
            };
            let right = if index + 1 == levels {
                10.0
            } else {
                f64::midpoint(f64::from(table[index]), f64::from(table[index + 1]))
            };
            // Independent composite midpoint quadrature of phi and x*phi.
            // Normalization of the standard normal density cancels in the ratio.
            let step = (right - left) / 4_096.0;
            let mut mass = 0.0;
            let mut moment = 0.0;
            for sample in 0..4_096_u32 {
                let point = (f64::from(sample) + 0.5).mul_add(step, left);
                let density = (-0.5 * point * point).exp();
                mass += density;
                moment += point * density;
            }
            assert!((moment / mass - f64::from(table[index])).abs() < 0.000_002);
        }
    }
}

#[test]
fn packed_dot_and_accumulation_match_independent_reconstruction() -> Result<()> {
    let query = input_row();
    for bits in [Bits::Three, Bits::Four] {
        let quantizer = Quantizer::scalar(bits, 91);
        let packed = record(bits, 2.0);
        let table = quantizer.centroids();
        let mut expected = [0.0; HEAD_DIM];
        let mut lanes = [0.0_f64; 4];
        for (index, value) in expected.iter_mut().enumerate() {
            let centroid = table[index % usize::from(bits.levels())];
            *value = centroid * 2.0;
            lanes[index % 4] += f64::from(query[index]) * f64::from(centroid);
        }
        let expected_dot =
            scalar::checked_f32(((lanes[0] + lanes[1]) + lanes[2]) + lanes[3])? * 2.0;
        assert_eq!(
            quantizer.score(&query, &packed)?.to_bits(),
            expected_dot.to_bits()
        );
        let mut accumulated = [0.0; HEAD_DIM];
        quantizer.accumulate(&packed, 1.0, &mut accumulated)?;
        assert_eq!(accumulated, expected);
        let mut decoded = [0.0; HEAD_DIM];
        quantizer.decode(&packed, &mut decoded)?;
        let mut reconstructed = [0.0; HEAD_DIM];
        quantizer.rotate(&decoded, &mut reconstructed)?;
        for (&actual, &expected) in reconstructed.iter().zip(&expected) {
            near(actual, expected, 0.000_004);
        }
    }
    Ok(())
}

#[test]
fn weighted_rows_can_accumulate_without_expanding_history() -> Result<()> {
    let quantizer = Quantizer::scalar(Bits::Four, 91);
    let first = record(Bits::Four, 2.0);
    let second = record(Bits::Four, 0.5);
    let mut actual = [0.0; HEAD_DIM];
    quantizer.accumulate(&first, 0.25, &mut actual)?;
    quantizer.accumulate(&second, 0.75, &mut actual)?;
    let mut expected = [0.0; HEAD_DIM];
    for (index, value) in expected.iter_mut().enumerate() {
        let centroid = tables::FOUR[index % 16];
        *value = 0.5 * centroid + 0.375 * centroid;
    }
    assert_eq!(actual, expected);
    Ok(())
}

#[test]
fn encode_preserves_norm_scale_and_has_bounded_fixture_error() -> Result<()> {
    let input = input_row();
    let norm_squared = input
        .iter()
        .map(|&value| f64::from(value).powi(2))
        .sum::<f64>();
    for (bits, max_relative_error) in [(Bits::Three, 0.12), (Bits::Four, 0.04)] {
        let quantizer = Quantizer::scalar(bits, 42);
        let mut packed = vec![0_u8; quantizer.encoded_len()];
        quantizer.encode(&input, &mut packed)?;
        let scale = f32::from_le_bytes(packed[..4].try_into()?);
        near(
            scale,
            scalar::checked_f32(norm_squared.sqrt() / 16.0)?,
            0.000_001,
        );
        let mut decoded = [0.0; HEAD_DIM];
        quantizer.decode(&packed, &mut decoded)?;
        let squared_error = input
            .iter()
            .zip(&decoded)
            .map(|(&left, &right)| (f64::from(left) - f64::from(right)).powi(2))
            .sum::<f64>();
        assert!(squared_error / norm_squared < max_relative_error);
    }
    Ok(())
}

#[test]
fn canonical_zero_is_exact_and_rejects_alternate_zero_records() -> Result<()> {
    for bits in [Bits::Three, Bits::Four] {
        let quantizer = Quantizer::scalar(bits, 42);
        let mut packed = vec![0xff_u8; quantizer.encoded_len()];
        quantizer.encode(&[-0.0; HEAD_DIM], &mut packed)?;
        assert!(packed.iter().all(|&value| value == 0));
        assert_eq!(quantizer.score(&[1.0; HEAD_DIM], &packed)?, 0.0);
        let mut accumulator = [7.0; HEAD_DIM];
        quantizer.accumulate(&packed, 100.0, &mut accumulator)?;
        assert_eq!(accumulator, [7.0; HEAD_DIM]);
        packed[4] = 1;
        assert!(quantizer.score(&[1.0; HEAD_DIM], &packed).is_err());
        packed[4] = 0;
        packed[..4].copy_from_slice(&(-0.0_f32).to_le_bytes());
        assert!(quantizer.score(&[1.0; HEAD_DIM], &packed).is_err());
    }
    Ok(())
}

#[test]
fn invalid_lengths_nonfinite_values_and_scale_underflow_reject_atomically() {
    let quantizer = Quantizer::scalar(Bits::Three, 42);
    let mut packed = vec![0x55_u8; quantizer.encoded_len()];
    let original = packed.clone();
    assert!(quantizer.encode(&[0.0; HEAD_DIM - 1], &mut packed).is_err());
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut input = [1.0; HEAD_DIM];
        input[2] = bad;
        assert!(quantizer.encode(&input, &mut packed).is_err());
        assert_eq!(packed, original);
    }
    let mut tiny = [0.0; HEAD_DIM];
    tiny[0] = f32::from_bits(1);
    assert!(quantizer.encode(&tiny, &mut packed).is_err());
    assert_eq!(packed, original);
    assert!(
        quantizer
            .encode(&[1.0; HEAD_DIM], &mut packed[..99])
            .is_err()
    );
    for bytes in [vec![], vec![0; 99], vec![0; 101]] {
        let mut output = [7.0; HEAD_DIM];
        assert!(quantizer.decode(&bytes, &mut output).is_err());
        assert_eq!(output, [7.0; HEAD_DIM]);
    }
}

#[test]
fn malformed_scales_and_overflow_do_not_partially_mutate_outputs() {
    let quantizer = Quantizer::scalar(Bits::Four, 91);
    for bad in [-1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY, f32::MAX] {
        let bytes = record(Bits::Four, bad);
        let mut output = [9.0; HEAD_DIM];
        assert!(quantizer.accumulate(&bytes, 1.0, &mut output).is_err());
        assert_eq!(output, [9.0; HEAD_DIM]);
    }
    let bytes = record(Bits::Four, 1.0);
    for bad in [f32::NAN, f32::INFINITY] {
        let mut output = [9.0; HEAD_DIM];
        assert!(quantizer.accumulate(&bytes, bad, &mut output).is_err());
        assert_eq!(output, [9.0; HEAD_DIM]);
        let mut query = [1.0; HEAD_DIM];
        query[1] = bad;
        assert!(quantizer.score(&query, &bytes).is_err());
    }
    let mut large = [f32::MAX; HEAD_DIM];
    let original = large;
    assert!(quantizer.inverse(&mut large).is_err());
    assert_eq!(large, original);
    let mut output = [9.0; HEAD_DIM];
    assert!(quantizer.rotate(&original, &mut output).is_err());
    assert_eq!(output, [9.0; HEAD_DIM]);
}

#[test]
fn automatic_and_forced_scalar_backends_share_packing_and_arithmetic() -> Result<()> {
    let input = input_row();
    for bits in [Bits::Three, Bits::Four] {
        let automatic = Quantizer::new(bits, 42);
        let scalar = Quantizer::scalar(bits, 42);
        let mut packed_auto = vec![0_u8; automatic.encoded_len()];
        let mut packed_scalar = packed_auto.clone();
        automatic.encode(&input, &mut packed_auto)?;
        scalar.encode(&input, &mut packed_scalar)?;
        assert_eq!(packed_auto, packed_scalar);
        let mut query_auto = [0.0; HEAD_DIM];
        let mut query_scalar = query_auto;
        automatic.rotate(&input, &mut query_auto)?;
        scalar.rotate(&input, &mut query_scalar)?;
        assert_eq!(query_auto, query_scalar);
        assert_eq!(
            automatic.score(&query_auto, &packed_auto)?.to_bits(),
            scalar.score(&query_scalar, &packed_scalar)?.to_bits()
        );
        let mut output_auto = [0.0; HEAD_DIM];
        let mut output_scalar = output_auto;
        automatic.accumulate(&packed_auto, 0.3, &mut output_auto)?;
        scalar.accumulate(&packed_scalar, 0.3, &mut output_scalar)?;
        assert_eq!(output_auto, output_scalar);
        automatic.inverse(&mut output_auto)?;
        scalar.inverse(&mut output_scalar)?;
        assert_eq!(output_auto, output_scalar);
    }
    Ok(())
}
