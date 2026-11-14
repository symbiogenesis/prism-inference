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

#[test]
fn ordered_centroid_search_preserves_exhaustive_rounding_and_first_ties() {
    for bits in [Bits::Three, Bits::Four] {
        let codec = Quantizer::scalar(bits, 91);
        let centroids = codec.centroids();
        let levels = bits.levels();
        assert!(
            centroids[..usize::from(levels)]
                .windows(2)
                .all(|pair| pair[0] < pair[1])
        );
        let check = |value: f32| {
            assert_eq!(
                scalar::nearest_ordered(value, centroids, levels),
                scalar::nearest(value, centroids, levels),
                "value bits {:08x}, precision {}",
                value.to_bits(),
                bits.bit_width(),
            );
        };
        for value in [
            0.0,
            -0.0,
            f32::from_bits(1),
            -f32::from_bits(1),
            f32::MIN_POSITIVE,
            -f32::MIN_POSITIVE,
            f32::MAX,
            -f32::MAX,
        ] {
            check(value);
        }
        for &centroid in &centroids[..usize::from(levels)] {
            for offset in 0..=4 {
                check(f32::from_bits(centroid.to_bits() + offset));
                check(f32::from_bits(centroid.to_bits() - offset));
            }
        }
        for pair in centroids[..usize::from(levels)].windows(2) {
            let midpoint = f32::midpoint(pair[0], pair[1]);
            for offset in 0..=4 {
                check(f32::from_bits(midpoint.to_bits().wrapping_add(offset)));
                check(f32::from_bits(midpoint.to_bits().wrapping_sub(offset)));
            }
        }
        // Sweep finite binary32 encodings across signs and exponents, including
        // large values whose rounded minimum spans several codebook entries.
        let mut state = 0x7f4a_7c15_u32;
        for _ in 0..65_536 {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let value = f32::from_bits(state);
            if value.is_finite() {
                check(value);
            }
        }
        for exponent in 1..=254 {
            for fraction in [0, 1, 0x003f_ffff, 0x007f_fffe, 0x007f_ffff] {
                let word = exponent << 23 | fraction;
                check(f32::from_bits(word));
                check(f32::from_bits(word | 0x8000_0000));
            }
        }
    }
}

#[test]
fn ordered_encoding_preserves_exhaustive_records_across_scale_extremes() -> Result<()> {
    let mut sparse = [0.0; HEAD_DIM];
    sparse[19] = -3.0;
    for bits in [Bits::Three, Bits::Four] {
        for fitted in [false, true] {
            let mut automatic = Quantizer::new(bits, 42);
            let mut exhaustive = Quantizer::scalar(bits, 42);
            if fitted {
                automatic = automatic.with_fitted_scale();
                exhaustive = exhaustive.with_fitted_scale();
            }
            for input in [
                input_row(),
                sparse,
                [-0.0; HEAD_DIM],
                [f32::MIN_POSITIVE; HEAD_DIM],
                [f32::MAX / 64.0; HEAD_DIM],
            ] {
                let mut actual = vec![0xff; automatic.encoded_len()];
                let mut expected = actual.clone();
                automatic.encode(&input, &mut actual)?;
                exhaustive.encode(&input, &mut expected)?;
                assert_eq!(actual, expected);
            }
        }
    }
    Ok(())
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

#[test]
fn x1_value_scan_preserves_record_rounding_and_rejects_late_failure() -> Result<()> {
    for bits in [Bits::Three, Bits::Four] {
        for scalar_only in [false, true] {
            let quantizer = if scalar_only {
                Quantizer::scalar(bits, 91)
            } else {
                Quantizer::new(bits, 91)
            };
            let mut scan = quantizer.value_accumulator()?;
            let mut expected = [0.0_f32; HEAD_DIM];
            let mut defensive = expected;
            for (scale, weight) in [(0.25, 0.125), (1.0, -0.5), (7.0, 0.375), (1.0, -0.0)] {
                let packed = record(bits, scale);
                scan.accumulate(&packed, weight)?;
                quantizer.accumulate(&packed, weight, &mut defensive)?;
                for (index, value) in expected.iter_mut().enumerate() {
                    let code = index % usize::from(bits.levels());
                    *value += (weight * scale) * quantizer.centroids()[code];
                }
            }
            let zero = vec![0; quantizer.encoded_len()];
            scan.accumulate(&zero, -0.0)?;
            assert_eq!(scan.finish()?.map(f32::to_bits), expected.map(f32::to_bits));
            assert_eq!(defensive.map(f32::to_bits), expected.map(f32::to_bits));
            let mut malformed_zero = zero;
            malformed_zero[4] = 1;
            for invalid in [
                Vec::new(),
                malformed_zero,
                record(bits, -0.0),
                record(bits, f32::NAN),
                record(bits, f32::INFINITY),
                record(bits, f32::MAX),
            ] {
                let mut scan = quantizer.value_accumulator()?;
                scan.accumulate(&record(bits, 1.0), 0.25)?;
                assert!(scan.accumulate(&invalid, 2.0).is_err());
                assert!(scan.accumulate(&record(bits, 1.0), 0.25).is_err());
                assert!(scan.finish().is_err());
            }
            let mut scan = quantizer.value_accumulator()?;
            scan.accumulate(&record(bits, 1.0), 0.25)?;
            assert!(scan.accumulate(&record(bits, 1.0), f32::NAN).is_err());
            assert!(scan.finish().is_err());
        }
    }
    Ok(())
}

#[test]
fn zero_weight_value_scan_preserves_signed_zero_cancellation_and_underflow() -> Result<()> {
    for bits in [Bits::Three, Bits::Four] {
        for quantizer in [Quantizer::scalar(bits, 91), Quantizer::new(bits, 91)] {
            let mut scan = quantizer.value_accumulator()?;
            let mut defensive = [0.0; HEAD_DIM];
            let minimum = f32::from_bits(1);
            for (scale, weight) in [
                (f32::MAX, 0.0),
                (f32::MAX, -0.0),
                (0.25, minimum),
                (0.25, -minimum),
                (minimum, 1.0),
                (minimum, -1.0),
                (1.0, -0.5),
                (1.0, -0.0),
                (0.25, minimum),
                (1.0, 0.5),
                (1.0, -0.0),
            ] {
                let packed = record(bits, scale);
                scan.accumulate(&packed, weight)?;
                quantizer.accumulate(&packed, weight, &mut defensive)?;
                // The independent public operation visits every coordinate,
                // including zero increments; cancellation must still yield +0.
                assert!(
                    defensive
                        .iter()
                        .all(|value| value.to_bits() != (-0.0_f32).to_bits())
                );
            }
            assert_eq!(
                scan.finish()?.map(f32::to_bits),
                defensive.map(f32::to_bits)
            );
            assert_eq!(defensive.map(f32::to_bits), [0.0_f32.to_bits(); HEAD_DIM]);

            // A caller-owned -0 row violates the private scan's invariant.
            // Public accumulation must still perform its signed additions.
            let mut negative_zero = [-0.0; HEAD_DIM];
            quantizer.accumulate(&record(bits, 1.0), -0.0, &mut negative_zero)?;
            for (index, value) in negative_zero.iter().enumerate() {
                let centroid = quantizer.centroids()[index % usize::from(bits.levels())];
                let expected = if centroid.is_sign_negative() {
                    0.0_f32
                } else {
                    -0.0_f32
                };
                assert_eq!(value.to_bits(), expected.to_bits());
            }
        }
    }
    Ok(())
}

#[test]
fn zero_weight_value_scan_still_validates_records_and_retains_poison() -> Result<()> {
    for bits in [Bits::Three, Bits::Four] {
        for quantizer in [Quantizer::scalar(bits, 91), Quantizer::new(bits, 91)] {
            let mut malformed_zero = vec![0; quantizer.encoded_len()];
            malformed_zero[4] = 1;
            for invalid in [
                Vec::new(),
                malformed_zero,
                record(bits, -0.0),
                record(bits, -1.0),
                record(bits, f32::NAN),
                record(bits, f32::INFINITY),
            ] {
                for weight in [0.0, -0.0, f32::from_bits(1)] {
                    let mut scan = quantizer.value_accumulator()?;
                    assert!(scan.accumulate(&invalid, weight).is_err());
                    assert!(scan.accumulate(&record(bits, 1.0), 0.0).is_err());
                    assert!(scan.finish().is_err());
                }
            }
            for weight in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                let mut scan = quantizer.value_accumulator()?;
                assert!(
                    scan.accumulate(&vec![0; quantizer.encoded_len()], weight)
                        .is_err()
                );
                assert!(scan.accumulate(&record(bits, 1.0), 0.0).is_err());
                assert!(scan.finish().is_err());
            }
        }
    }
    Ok(())
}

#[test]
fn scan_value_overflow_rejects_finite_products_and_preserves_public_output() -> Result<()> {
    for bits in [Bits::Three, Bits::Four] {
        for quantizer in [Quantizer::scalar(bits, 91), Quantizer::new(bits, 91)] {
            let scale = f32::MAX / 4.0;
            let mut packed = vec![0xff; quantizer.encoded_len()];
            packed[..4].copy_from_slice(&scale.to_le_bytes());
            let mut scan = quantizer.value_accumulator()?;
            // The largest centroid lies between two and four: each product
            // is finite, but adding a second identical product overflows.
            let largest = quantizer.centroids()[usize::from(bits.levels() - 1)];
            assert!(largest > 2.0 && largest < 4.0);
            assert!((scale * largest).is_finite());
            scan.accumulate(&packed, 1.0)?;
            assert!(scan.accumulate(&packed, 1.0).is_err());
            assert!(scan.accumulate(&record(bits, 1.0), 0.0).is_err());
            assert!(scan.finish().is_err());

            // Only the final coordinate fails, after all earlier updates;
            // the defensive public helper still publishes none of them.
            let mut output = [0.0; HEAD_DIM];
            output[HEAD_DIM - 1] = f32::MAX;
            let original = output.map(f32::to_bits);
            assert!(quantizer.accumulate(&packed, 1.0, &mut output).is_err());
            assert_eq!(output.map(f32::to_bits), original);
        }
    }
    Ok(())
}

#[test]
fn fused_encode_validation_preserves_exact_norm_and_rejects_nonfinite_tail() -> Result<()> {
    let mut input = [0.0; HEAD_DIM];
    input[0] = 3.0;
    input[HEAD_DIM - 1] = -4.0;
    for bits in [Bits::Three, Bits::Four] {
        for quantizer in [Quantizer::scalar(bits, 91), Quantizer::new(bits, 91)] {
            let mut packed = vec![0xff; quantizer.encoded_len()];
            quantizer.encode(&input, &mut packed)?;
            assert_eq!(&packed[..4], &(5.0_f32 / 16.0).to_le_bytes());
            for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                let mut malformed = input;
                malformed[HEAD_DIM - 1] = invalid;
                let original = packed.clone();
                assert!(quantizer.encode(&malformed, &mut packed).is_err());
                assert_eq!(packed, original);
            }
        }
    }
    Ok(())
}

#[test]
fn x1_prepared_kv_scores_preserve_order_records_and_failure_precedence() -> Result<()> {
    let query = input_row();
    for bits in [Bits::Three, Bits::Four] {
        for scalar_only in [false, true] {
            let quantizer = if scalar_only {
                Quantizer::scalar(bits, 91)
            } else {
                Quantizer::new(bits, 91)
            };
            let prepared = quantizer.prepare_query(&query)?;
            for scale in [f32::from_bits(1), 0.25, 1.0, 7.0] {
                let packed = record(bits, scale);
                let mut lanes = [0.0_f64; 4];
                for (index, &coordinate) in query.iter().enumerate() {
                    let code = index % usize::from(bits.levels());
                    lanes[index % 4] +=
                        f64::from(coordinate) * f64::from(quantizer.centroids()[code]);
                }
                let expected =
                    scalar::checked_f32(((lanes[0] + lanes[1]) + lanes[2]) + lanes[3])? * scale;
                assert_eq!(prepared.score(&packed)?.to_bits(), expected.to_bits());
            }
            let mut zero = vec![0; quantizer.encoded_len()];
            assert_eq!(prepared.score(&zero)?.to_bits(), 0.0_f32.to_bits());
            zero[4] = 1;
            assert!(prepared.score(&zero).is_err());
            for scale in [-0.0, -1.0, f32::INFINITY, f32::NAN] {
                assert!(prepared.score(&record(bits, scale)).is_err());
            }
            assert!(prepared.score(&[]).is_err());
            let mut invalid_query = query;
            invalid_query[255] = f32::NAN;
            let error = quantizer
                .score(&invalid_query, &[])
                .err()
                .ok_or_else(|| crate::invalid("invalid query must fail before invalid record"))?;
            assert!(
                error
                    .to_string()
                    .contains("row must contain only finite values")
            );
        }
    }
    Ok(())
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

fn rotated_reconstruction_error(
    quantizer: &Quantizer,
    input: &[f32; HEAD_DIM],
    baseline: &[u8],
    packed: &[u8],
) -> Result<f64> {
    let norm_scale = f32::from_le_bytes(baseline[..4].try_into()?);
    let normalized =
        input.map(|value| scalar::checked_f32(f64::from(value) / f64::from(norm_scale)));
    let mut normalized_row = [0.0; HEAD_DIM];
    for (value, result) in normalized_row.iter_mut().zip(normalized) {
        *value = result?;
    }
    let mut target = [0.0; HEAD_DIM];
    quantizer.rotate(&normalized_row, &mut target)?;
    let mut reconstructed = [0.0; HEAD_DIM];
    // Exercise the unchanged packed reader, independently of the fitter.
    quantizer.accumulate(packed, 1.0, &mut reconstructed)?;
    Ok(target
        .iter()
        .zip(reconstructed)
        .map(|(&value, decoded)| {
            (f64::from(value) * f64::from(norm_scale) - f64::from(decoded)).powi(2)
        })
        .sum())
}

#[test]
fn fitted_scale_preserves_codes_identity_and_improves_rotated_fixture_error() -> Result<()> {
    let mut sparse = [0.0; HEAD_DIM];
    sparse[19] = 3.0;
    let fixtures = [
        input_row(),
        sparse,
        [1.0; HEAD_DIM],
        input_row().map(|value| value * 1.0e-39),
        input_row().map(|value| value * 1.0e30),
    ];
    let mut improvements = 0;
    for bits in [Bits::Three, Bits::Four] {
        let baseline_codec = Quantizer::scalar(bits, 42);
        let fitted = baseline_codec.clone().with_fitted_scale();
        let automatic = Quantizer::new(bits, 42).with_fitted_scale();
        assert_eq!(baseline_codec.identity().version, 1);
        assert_eq!(fitted.identity().version, 2);
        assert_ne!(baseline_codec.identity(), fitted.identity());
        assert_eq!(fitted.identity(), automatic.identity());
        for input in &fixtures {
            let mut baseline = vec![0; baseline_codec.encoded_len()];
            let mut actual = vec![0; fitted.encoded_len()];
            let mut native = vec![0; automatic.encoded_len()];
            baseline_codec.encode(input, &mut baseline)?;
            fitted.encode(input, &mut actual)?;
            automatic.encode(input, &mut native)?;
            assert_eq!(actual, native);
            assert_eq!(&actual[4..], &baseline[4..]);
            let before =
                rotated_reconstruction_error(&baseline_codec, input, &baseline, &baseline)?;
            let after = rotated_reconstruction_error(&baseline_codec, input, &baseline, &actual)?;
            assert!(after <= before * (1.0 + 1.0e-12), "{after} > {before}");
            improvements += usize::from(after < before);
        }
    }
    assert!(
        improvements > 0,
        "fitted scaling must improve a nontrivial fixture"
    );
    Ok(())
}

#[test]
fn fitted_scale_zero_and_invalid_encodes_preserve_atomicity() -> Result<()> {
    for bits in [Bits::Three, Bits::Four] {
        let codec = Quantizer::scalar(bits, 42).with_fitted_scale();
        let mut packed = vec![0xff; codec.encoded_len()];
        codec.encode(&[-0.0; HEAD_DIM], &mut packed)?;
        assert!(packed.iter().all(|&byte| byte == 0));
        let mut invalid = [1.0; HEAD_DIM];
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            invalid[19] = value;
            assert!(codec.encode(&invalid, &mut packed).is_err());
            assert!(packed.iter().all(|&byte| byte == 0));
        }
        let mut underflow = [0.0; HEAD_DIM];
        underflow[0] = f32::from_bits(1);
        assert!(codec.encode(&underflow, &mut packed).is_err());
        assert!(codec.encode(&[0.0; HEAD_DIM - 1], &mut packed).is_err());
        assert!(packed.iter().all(|&byte| byte == 0));
    }
    Ok(())
}

#[test]
fn fitted_scale_keeps_baseline_for_unusable_or_unchanged_candidates() {
    let codes = [0xff; HEAD_DIM / 2];
    for (row, scale) in [
        ([tables::FOUR[15]; HEAD_DIM], 1.0),
        ([16.0; HEAD_DIM], f32::MAX),
        ([0.01; HEAD_DIM], f32::from_bits(1)),
    ] {
        let selected = super::fitted_scale::select(&row, &codes, Bits::Four, &tables::FOUR, scale);
        assert_eq!(selected.to_bits(), scale.to_bits());
    }
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
