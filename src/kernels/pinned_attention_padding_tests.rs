//! Full physical-padding controls retained independently of the pruned loops.

use bonsai_ternary::half_to_float;

use super::{VALUE_TILE, dot, softmax, value_tile};

fn full_padding_dot(values: impl Iterator<Item = f32>, weights: &[f32]) -> f32 {
    let physical = weights.len().div_ceil(256) * 256;
    let values = values.chain(std::iter::repeat(0.0));
    let weights = weights.iter().copied().chain(std::iter::repeat(0.0));
    let mut registers = [[0.0_f32; 4]; 4];
    for (index, (value, weight)) in values.zip(weights).take(physical).enumerate() {
        let accumulator = &mut registers[index % 16 / 4][index % 4];
        *accumulator = value.mul_add(weight, *accumulator);
    }
    let lanes: [f32; 4] = std::array::from_fn(|lane| {
        (registers[0][lane] + registers[2][lane]) + (registers[1][lane] + registers[3][lane])
    });
    (lanes[0] + lanes[1]) + (lanes[2] + lanes[3])
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::as_conversions,
    reason = "The frozen physical-padding control rounds the native binary64 reciprocal to f32."
)]
fn full_padding_softmax(scores: &mut [f32]) {
    let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let physical = scores.len().div_ceil(256) * 256;
    let mut total = 0.0_f64;
    for offset in (0..physical).step_by(4) {
        let mut group = [0.0; 4];
        for (lane, output) in group.iter_mut().enumerate() {
            if let Some(score) = scores.get_mut(offset + lane) {
                *score = super::super::pinned_activations::vector_exp(*score - maximum);
                *output = *score;
            }
        }
        total += f64::from((group[0] + group[1]) + (group[2] + group[3]));
    }
    let inverse = (1.0_f64 / total) as f32;
    for score in scores {
        *score *= inverse;
    }
}

#[test]
fn pruned_dot_padding_matches_full_physical_rows_bitwise() -> crate::Result<()> {
    let pattern = [
        0.0,
        -0.0,
        1.0,
        -1.0,
        f32::MIN_POSITIVE,
        f32::from_bits(1),
        -0.25,
    ];
    for length in (1..=272).chain([511, 512, 513, 8191, 8192]) {
        for shift in 0..pattern.len() {
            let values: Vec<_> = (0..length)
                .map(|index| pattern[(index + shift) % pattern.len()])
                .collect();
            let weights: Vec<_> = (0..length)
                .map(|index| pattern[(index * 3 + shift) % pattern.len()])
                .collect();
            assert_eq!(
                dot(values.iter().copied(), &weights)?.to_bits(),
                full_padding_dot(values.into_iter(), &weights).to_bits(),
                "length {length}, shift {shift}"
            );
        }
    }
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, f32::MAX] {
        assert!(dot([value].into_iter(), &[2.0]).is_err());
    }
    assert!(dot([].into_iter(), &[1.0]).is_err());
    assert!(dot([1.0, 2.0].into_iter(), &[1.0]).is_err());
    assert!(dot(std::iter::repeat(1.0), &vec![1.0; 8193]).is_err());
    Ok(())
}

#[test]
fn pruned_value_padding_matches_full_rows_at_every_lane_boundary() -> crate::Result<()> {
    let pattern = [
        0_u16, 0x8000, 0x0001, 0x8001, 0x3c00, 0xbc00, 0x7bff, 0xfbff,
    ];
    for length in [
        1, 15, 16, 17, 239, 240, 241, 255, 256, 257, 271, 272, 8191, 8192,
    ] {
        for width in [1, 7, VALUE_TILE] {
            let rows: Vec<Vec<u16>> = (0..length)
                .map(|index| {
                    (0..width + 3)
                        .map(|channel| pattern[(index * 3 + channel) % pattern.len()])
                        .collect()
                })
                .collect();
            let current: Vec<_> = rows[length - 1][3..]
                .iter()
                .copied()
                .map(half_to_float)
                .collect();
            let weights: Vec<_> = (0..length)
                .map(|index| half_to_float(pattern[index % pattern.len()]))
                .collect();
            let mut output = vec![f32::NAN; width];
            value_tile(
                rows[..length - 1].iter().map(Vec::as_slice),
                &current,
                &weights,
                3,
                &mut output,
            )?;
            for (channel, actual) in output.into_iter().enumerate() {
                let values = rows.iter().map(|row| half_to_float(row[channel + 3]));
                assert_eq!(
                    actual.to_bits(),
                    full_padding_dot(values, &weights).to_bits(),
                    "length {length}, width {width}, channel {channel}"
                );
            }
        }
    }
    Ok(())
}

#[test]
fn pruned_softmax_padding_matches_full_physical_groups_bitwise() -> crate::Result<()> {
    let pattern = [
        -1000.0,
        -90.0,
        -1.0,
        -f32::from_bits(1),
        -0.0,
        0.0,
        1.0,
        90.0,
        1000.0,
    ];
    for length in (1..=272).chain([511, 512, 513, 8191, 8192]) {
        for shift in 0..pattern.len() {
            let mut actual: Vec<_> = (0..length)
                .map(|index| pattern[(index + shift) % pattern.len()])
                .collect();
            let mut expected = actual.clone();
            full_padding_softmax(&mut expected);
            softmax(&mut actual)?;
            assert!(
                actual
                    .iter()
                    .zip(expected)
                    .all(|(actual, expected)| actual.to_bits() == expected.to_bits()),
                "length {length}, shift {shift}"
            );
        }
    }
    Ok(())
}
