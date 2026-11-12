//! Independent single-head arithmetic, cache visitation and failure controls.

use std::cell::Cell;

use bonsai_ternary::half_to_float;

use super::{dot, dot_pair, value_tile_gated};

fn outputs(values: &mut [Vec<f32>; 6]) -> [&mut [f32]; 6] {
    let [first, second, third, fourth, fifth, sixth] = values;
    [first, second, third, fourth, fifth, sixth]
}

#[test]
fn six_head_scores_match_independent_native_lanes_and_tails() -> crate::Result<()> {
    let pattern = [0, 0x8000, 1, 0x8001, 0x3555, 0xb555, 0x3c00, 0xbc00];
    for width in [1, 3, 4, 7, 8, 15, 16, 17, 127, 128, 255, 256] {
        let key: Vec<u16> = (0..width).map(|index| pattern[index % 8]).collect();
        let queries: [Vec<f32>; 6] = std::array::from_fn(|head| {
            (0..width)
                .map(|index| half_to_float(pattern[(index + head) % 8]))
                .collect()
        });
        let actual = dot(&key, queries.each_ref().map(Vec::as_slice))?;
        for (actual, query) in actual.into_iter().zip(queries) {
            let expected = super::super::dot(key.iter().copied().map(half_to_float), &query)?;
            assert_eq!(actual.to_bits(), expected.to_bits());
        }
    }
    Ok(())
}

#[test]
fn six_head_aligned_scores_keep_native_vector_reduction_bits() -> crate::Result<()> {
    // The scalar iterator helper retains the old independent four-register
    // finish. Include aligned masked cycles, no padding, and scalar tails.
    let key_pattern = [
        0x0000, 0x8000, 0x0001, 0x8001, 0x3c00, 0xbc00, 0x7bff, 0xfbff,
    ];
    let query_pattern = [
        0.0,
        -0.0,
        f32::from_bits(1),
        -f32::from_bits(1),
        f32::from_bits(0x3f80_0001),
        -f32::from_bits(0x3f80_0001),
        0.25,
        -0.25,
    ];
    for width in [
        1, 7, 15, 16, 17, 31, 32, 64, 128, 240, 255, 256, 257, 272, 512,
    ] {
        for shift in 0..key_pattern.len() {
            let key: Vec<_> = (0..width)
                .map(|index| key_pattern[(index + shift) % key_pattern.len()])
                .collect();
            let queries: [Vec<f32>; 12] = std::array::from_fn(|head| {
                (0..width)
                    .map(|index| query_pattern[(index * 3 + head + shift) % query_pattern.len()])
                    .collect()
            });
            let first = std::array::from_fn(|head| queries[head].as_slice());
            let second = std::array::from_fn(|head| queries[head + 6].as_slice());
            let paired = dot_pair(&key, [first, second])?;
            assert_eq!(
                dot(&key, first)?.map(f32::to_bits),
                paired[0].map(f32::to_bits)
            );
            for (head, query) in queries.iter().enumerate() {
                let expected = super::super::dot(key.iter().copied().map(half_to_float), query)?;
                assert_eq!(
                    paired[head / 6][head % 6].to_bits(),
                    expected.to_bits(),
                    "width {width}, shift {shift}, head {head}"
                );
            }
        }
    }
    for width in [16, 32, 128, 256] {
        let key = vec![0x3c00; width];
        let mut queries: [Vec<f32>; 6] = std::array::from_fn(|_| vec![1.0; width]);
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, f32::MAX] {
            queries[5].fill(value);
            assert!(dot(&key, queries.each_ref().map(Vec::as_slice)).is_err());
        }
    }
    Ok(())
}

#[test]
fn x1_attention_fixed_lane_scores_match_independent_six_and_twelve_head_trees() -> crate::Result<()>
{
    // Cover every transition between the sixteen-column body, its optional
    // eight-column vector tail, scalar tail and masked temporal lanes. The
    // iterator oracle retains its independent scalar four-register tree.
    let keys = [
        0x0000, 0x8000, 0x0001, 0x8001, 0x3555, 0xb555, 0x7bff, 0xfbff,
    ];
    let query_values = [
        0.0,
        -0.0,
        f32::from_bits(1),
        -f32::from_bits(1),
        f32::from_bits(0x3f80_0001),
        -f32::from_bits(0x3f80_0001),
        1.0e20,
        -1.0e20,
    ];
    for width in (1..=33).chain([63, 64, 65, 127, 128, 129, 239, 240, 241, 255, 256, 257]) {
        for shift in 0..keys.len() {
            let key: Vec<_> = (0..width)
                .map(|index| keys[(index + shift) % keys.len()])
                .collect();
            let queries: [Vec<f32>; 12] = std::array::from_fn(|head| {
                (0..width)
                    .map(|index| query_values[(index * 3 + head + shift) % query_values.len()])
                    .collect()
            });
            let borrowed = queries.each_ref().map(Vec::as_slice);
            let first = std::array::from_fn(|head| borrowed[head]);
            let second = std::array::from_fn(|head| borrowed[head + 6]);
            let actual = dot_pair(&key, [first, second])?;
            assert_eq!(
                dot(&key, first)?.map(f32::to_bits),
                actual[0].map(f32::to_bits)
            );
            for (head, query) in borrowed.into_iter().enumerate() {
                let expected = super::super::dot(key.iter().copied().map(half_to_float), query)?;
                assert_eq!(
                    actual[head / 6][head % 6].to_bits(),
                    expected.to_bits(),
                    "width {width}, shift {shift}, head {head}"
                );
            }
        }
    }
    for width in [8, 15, 16, 17, 24, 31, 32, 33, 255, 256, 257] {
        let key = vec![0x3c00; width];
        let mut queries: [Vec<f32>; 12] = std::array::from_fn(|_| vec![1.0; width]);
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, f32::MAX] {
            queries[11].fill(value);
            let borrowed = queries.each_ref().map(Vec::as_slice);
            assert!(
                dot_pair(
                    &key,
                    [
                        std::array::from_fn(|head| borrowed[head]),
                        std::array::from_fn(|head| borrowed[head + 6]),
                    ],
                )
                .is_err()
            );
        }
    }
    Ok(())
}

#[test]
fn six_head_values_match_independent_channels_and_visit_each_row_once() -> crate::Result<()> {
    let pattern = [0, 0x8000, 1, 0x8001, 0x3555, 0xb555, 0x3c00, 0xbc00];
    for count in [1, 2, 15, 16, 17, 64, 65, 255, 256, 257] {
        for width in [1, 3, 4, 7, 8, 31, 32, 63, 64] {
            let offset = 3;
            let rows: Vec<Vec<u16>> = (0..count - 1)
                .map(|token| {
                    (0..offset + width)
                        .map(|c| pattern[(token + c) % 8])
                        .collect()
                })
                .collect();
            let current: Vec<f32> = (0..width).map(|c| half_to_float(pattern[c % 8])).collect();
            let weights: [Vec<f32>; 6] = std::array::from_fn(|head| {
                (0..count)
                    .map(|token| half_to_float(pattern[(token + head) % 8]))
                    .collect()
            });
            let gates: [Vec<f32>; 6] = std::array::from_fn(|head| {
                (0..width)
                    .map(|c| [-100.0, -4.0, -0.0, 0.0, 1.0, 100.0][(c + head) % 6])
                    .collect()
            });
            let visits = Cell::new(0);
            let mut actual = std::array::from_fn(|_| vec![19.0; width]);
            value_tile_gated(
                rows.iter().map(|row| {
                    visits.set(visits.get() + 1);
                    row.as_slice()
                }),
                &current,
                weights.each_ref().map(Vec::as_slice),
                offset,
                outputs(&mut actual),
                gates.each_ref().map(Vec::as_slice),
            )?;
            assert_eq!(visits.get(), count - 1);
            for (head, output) in actual.iter().enumerate() {
                for (channel, &output) in output.iter().enumerate() {
                    let expected = super::super::dot(
                        rows.iter()
                            .map(|row| half_to_float(row[offset + channel]))
                            .chain(std::iter::once(current[channel])),
                        &weights[head],
                    )? * (1.0 / (1.0 + (-gates[head][channel]).exp()));
                    assert_eq!(output.to_bits(), expected.to_bits());
                }
            }
        }
    }
    Ok(())
}

#[test]
fn six_head_values_reject_invalid_geometry_and_late_failure_atomically() {
    let gates = [0.0; 4];
    let mut actual: [Vec<f32>; 6] = std::array::from_fn(|_| vec![19.0; 4]);
    let initial = actual.clone();
    let weights = [1.0; 2];
    let rows = [[0x3c00; 4]];
    assert!(
        value_tile_gated(
            rows.iter().map(<[u16; 4]>::as_slice),
            &[1.0; 4],
            [&weights; 6],
            usize::MAX,
            outputs(&mut actual),
            [&gates; 6],
        )
        .is_err()
    );
    assert_eq!(actual, initial);
    assert!(
        value_tile_gated(
            std::iter::empty(),
            &[1.0; 4],
            [&weights; 6],
            0,
            outputs(&mut actual),
            [&gates; 6],
        )
        .is_err()
    );
    assert_eq!(actual, initial);
    assert!(
        value_tile_gated(
            rows.iter().map(<[u16; 4]>::as_slice),
            &[1.0; 4],
            [&[1.0]; 6],
            0,
            outputs(&mut actual),
            [&gates; 6],
        )
        .is_err()
    );
    assert_eq!(actual, initial);
    let bad = [f32::NAN, 1.0];
    let mut head_weights = [&weights[..]; 6];
    head_weights[5] = &bad;
    assert!(
        value_tile_gated(
            rows.iter().map(<[u16; 4]>::as_slice),
            &[1.0; 4],
            head_weights,
            0,
            outputs(&mut actual),
            [&gates; 6],
        )
        .is_err()
    );
    assert_eq!(actual, initial);
    let short_gate = [0.0; 3];
    let mut gate_rows = [&gates[..]; 6];
    gate_rows[5] = &short_gate;
    assert!(
        value_tile_gated(
            rows.iter().map(<[u16; 4]>::as_slice),
            &[1.0; 4],
            [&weights; 6],
            0,
            outputs(&mut actual),
            gate_rows,
        )
        .is_err()
    );
    assert_eq!(actual, initial);
    assert!(dot(&[0x3c00; 4], [&[0.0; 3]; 6]).is_err());
    assert!(dot(&[0x7c00; 4], [&[1.0; 4]; 6]).is_err());
}

#[test]
fn six_head_paired_scores_match_independent_native_lanes_and_reject_bad_rows() -> crate::Result<()>
{
    let pattern = [0, 0x8000, 1, 0x8001, 0x3555, 0xb555, 0x3c00, 0xbc00];
    for width in [1, 3, 4, 7, 8, 15, 16, 17, 127, 128, 255, 256] {
        let key: Vec<u16> = (0..width).map(|index| pattern[index % 8]).collect();
        let queries: [[Vec<f32>; 6]; 2] = std::array::from_fn(|token| {
            std::array::from_fn(|head| {
                (0..width)
                    .map(|index| half_to_float(pattern[(index + head + token * 3) % 8]))
                    .collect()
            })
        });
        let borrowed = queries
            .each_ref()
            .map(|rows| rows.each_ref().map(Vec::as_slice));
        let actual = dot_pair(&key, borrowed)?;
        for (actual, queries) in actual.into_iter().zip(&queries) {
            for (actual, query) in actual.into_iter().zip(queries) {
                let expected = super::super::dot(key.iter().copied().map(half_to_float), query)?;
                assert_eq!(actual.to_bits(), expected.to_bits());
            }
        }
        let mut bad = borrowed;
        bad[1][5] = &queries[1][5][..width - 1];
        assert!(dot_pair(&key, bad).is_err());
    }
    assert!(dot_pair(&[], [[&[]; 6]; 2]).is_err());
    let mut queries = [[&[1.0_f32][..]; 6]; 2];
    queries[1][5] = &[f32::NAN];
    assert!(dot_pair(&[0x3c00], queries).is_err());
    Ok(())
}
