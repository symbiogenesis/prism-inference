//! Independent native-dot, shared-cache visit and publication controls.

use std::cell::Cell;

use bonsai_ternary::half_to_float;

use super::{scalar, value_pair_tile_gated};

fn outputs(values: &mut [[Vec<f32>; 6]; 2]) -> [[&mut [f32]; 6]; 2] {
    values.each_mut().map(|rows| {
        let [first, second, third, fourth, fifth, sixth] = rows;
        [first.as_mut_slice(), second, third, fourth, fifth, sixth]
    })
}

#[test]
fn paired_values_match_native_dot_and_share_common_cache_rows() -> crate::Result<()> {
    let pattern = [0, 0x8000, 1, 0x8001, 0x3555, 0xb555, 0x3c00, 0xbc00];
    for count in [1, 2, 15, 16, 17, 63, 64, 65, 127] {
        for width in [1, 3, 4, 7, 8, 31, 32] {
            let offset = 3;
            let rows: Vec<Vec<u16>> = (0..count)
                .map(|token| {
                    (0..offset + width)
                        .map(|channel| pattern[(token + channel) % 8])
                        .collect()
                })
                .collect();
            let current: [Vec<f32>; 2] = std::array::from_fn(|token| {
                (0..width)
                    .map(|channel| [0.33337, -0.251, -0.0, 0.501][(channel + token) % 4])
                    .collect()
            });
            let weights: [[Vec<f32>; 6]; 2] = std::array::from_fn(|token| {
                std::array::from_fn(|head| {
                    (0..count + token)
                        .map(|position| half_to_float(pattern[(position + head + token) % 8]))
                        .collect()
                })
            });
            let gates: [[Vec<f32>; 6]; 2] = std::array::from_fn(|token| {
                std::array::from_fn(|head| {
                    (0..width)
                        .map(|channel| {
                            [-100.0, -4.0, -0.0, 0.0, 1.0, 100.0][(channel + head + token) % 6]
                        })
                        .collect()
                })
            });
            let borrowed_weights = weights.each_ref().map(|row| row.each_ref().map(Vec::as_slice));
            let borrowed_gates = gates.each_ref().map(|row| row.each_ref().map(Vec::as_slice));
            let mut actual: [[Vec<f32>; 6]; 2] =
                std::array::from_fn(|_| std::array::from_fn(|_| vec![19.0; width]));
            let visits = Cell::new(0);
            value_pair_tile_gated(
                rows.iter().map(|row| {
                    visits.set(visits.get() + 1);
                    row.as_slice()
                }),
                current.each_ref().map(Vec::as_slice),
                borrowed_weights,
                offset,
                outputs(&mut actual),
                borrowed_gates,
            )?;
            assert_eq!(visits.get(), count);
            let mut scalar_output = actual.clone();
            scalar(
                rows.iter().map(Vec::as_slice),
                current.each_ref().map(Vec::as_slice),
                borrowed_weights,
                &(offset..offset + width),
                outputs(&mut scalar_output),
                borrowed_gates,
            )?;
            for (token, actual) in actual.iter().enumerate() {
                for (head, actual) in actual.iter().enumerate() {
                    for (channel, actual) in actual.iter().enumerate() {
                        let expected = super::super::super::dot(
                            rows.iter()
                                .take(count - 1 + token)
                                .map(|row| half_to_float(row[offset + channel]))
                                .chain(std::iter::once(current[token][channel])),
                            &weights[token][head],
                        )? * (1.0 / (1.0 + (-gates[token][head][channel]).exp()));
                        assert_eq!(actual.to_bits(), expected.to_bits());
                        assert_eq!(
                            scalar_output[token][head][channel].to_bits(),
                            expected.to_bits()
                        );
                    }
                }
            }
        }
    }
    Ok(())
}

#[test]
fn paired_values_reject_malformed_and_late_second_token_failure_before_publication() {
    let weights = [1.0; 16];
    let second_weights = [1.0; 17];
    let borrowed_weights = [[&weights[..]; 6], [&second_weights[..]; 6]];
    let gates = [[&[0.0; 4][..]; 6]; 2];
    let rows = [[0x3c00; 4]; 17];
    let mut actual: [[Vec<f32>; 6]; 2] =
        std::array::from_fn(|_| std::array::from_fn(|_| vec![19.0; 4]));
    let retained = actual.clone();
    for count in [0, 14, 15, 17] {
        assert!(
            value_pair_tile_gated(
                rows[..count].iter().map(<[u16; 4]>::as_slice),
                [&[1.0; 4]; 2],
                borrowed_weights,
                0,
                outputs(&mut actual),
                gates,
            )
            .is_err()
        );
        assert_eq!(actual, retained);
    }
    for offset in [1, usize::MAX] {
        assert!(
            value_pair_tile_gated(
                rows[..16].iter().map(<[u16; 4]>::as_slice),
                [&[1.0; 4]; 2],
                borrowed_weights,
                offset,
                outputs(&mut actual),
                gates,
            )
            .is_err()
        );
        assert_eq!(actual, retained);
    }
    let mut invalid_weights = borrowed_weights;
    invalid_weights[1][5] = &weights;
    assert!(
        value_pair_tile_gated(
            rows[..16].iter().map(<[u16; 4]>::as_slice),
            [&[1.0; 4]; 2],
            invalid_weights,
            0,
            outputs(&mut actual),
            gates,
        )
        .is_err()
    );
    assert_eq!(actual, retained);
    let mut bad = second_weights;
    bad[16] = f32::NAN;
    let mut invalid_weights = borrowed_weights;
    invalid_weights[1][5] = &bad;
    assert!(
        value_pair_tile_gated(
            rows[..16].iter().map(<[u16; 4]>::as_slice),
            [&[1.0; 4]; 2],
            invalid_weights,
            0,
            outputs(&mut actual),
            gates,
        )
        .is_err()
    );
    assert_eq!(actual, retained);
}
