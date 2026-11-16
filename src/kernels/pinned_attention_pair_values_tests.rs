//! Independent native-dot, shared-cache visit and publication controls.

use std::cell::Cell;

use bonsai_ternary::half_to_float;

use super::{
    scalar, scalar_from_scores, scalar_packed, value_pair_tile_from_scores_gated,
    value_pair_tile_gated, value_pair_tile_packed_gated,
};

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
            let borrowed_weights = weights
                .each_ref()
                .map(|row| row.each_ref().map(Vec::as_slice));
            let borrowed_gates = gates
                .each_ref()
                .map(|row| row.each_ref().map(Vec::as_slice));
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

fn build_softmax_rows(
    count: usize,
    round_f16: bool,
) -> crate::Result<[Vec<Vec<f32>>; 2]> {
    let raw_pattern = [-2.5_f32, -0.5, 0.0, 0.75, 1.5, -1.25, 0.375];
    let mut out = [Vec::with_capacity(6), Vec::with_capacity(6)];
    for (token, token_rows) in out.iter_mut().enumerate() {
        for head in 0..6 {
            let mut row: Vec<f32> = (0..count + token)
                .map(|pos| raw_pattern[(pos * 3 + head + token * 2) % raw_pattern.len()])
                .collect();
            if round_f16 {
                super::super::super::softmax_round_f16(&mut row)?;
            } else {
                super::super::super::softmax(&mut row)?;
            }
            token_rows.push(row);
        }
    }
    Ok(out)
}

fn assert_pair_outputs_eq(actual: &[[Vec<f32>; 6]; 2], expected: &[[Vec<f32>; 6]; 2]) {
    for (act_tok, exp_tok) in actual.iter().zip(expected) {
        for (act_head, exp_head) in act_tok.iter().zip(exp_tok) {
            for (act_val, exp_val) in act_head.iter().zip(exp_head) {
                assert_eq!(act_val.to_bits(), exp_val.to_bits());
            }
        }
    }
}

struct PairedFixture {
    rows: Vec<Vec<u16>>,
    current: [Vec<f32>; 2],
    normalized: [Vec<Vec<f32>>; 2],
    rounded: [Vec<Vec<f32>>; 2],
    packed0: Vec<[u16; 6]>,
    packed1: Vec<[u16; 6]>,
    gates: [[Vec<f32>; 6]; 2],
}

fn build_paired_fixture(count: usize, width: usize) -> crate::Result<PairedFixture> {
    let pattern = [0, 0x8000, 1, 0x8001, 0x3555, 0xb555, 0x3c00, 0xbc00];
    let offset = 2;
    let rows: Vec<Vec<u16>> = (0..count)
        .map(|token| {
            (0..offset + width)
                .map(|channel| pattern[(token + channel) % pattern.len()])
                .collect()
        })
        .collect();
    let current: [Vec<f32>; 2] = std::array::from_fn(|token| {
        (0..width)
            .map(|channel| [0.33337, -0.251, -0.0, 0.501][(channel + token) % 4])
            .collect()
    });
    let normalized = build_softmax_rows(count, false)?;
    let rounded = build_softmax_rows(count, true)?;
    let mut packed0 = vec![[0_u16; 6]; count];
    let mut packed1 = vec![[0_u16; 6]; count + 1];
    super::super::super::pack_f16_rows(
        std::array::from_fn(|head| normalized[0][head].as_slice()),
        &mut packed0,
    )?;
    super::super::super::pack_f16_rows(
        std::array::from_fn(|head| normalized[1][head].as_slice()),
        &mut packed1,
    )?;
    let gates: [[Vec<f32>; 6]; 2] = std::array::from_fn(|token| {
        std::array::from_fn(|head| {
            (0..width)
                .map(|channel| [-100.0, -4.0, -0.0, 0.0, 1.0, 100.0][(channel + head + token) % 6])
                .collect()
        })
    });
    Ok(PairedFixture {
        rows,
        current,
        normalized,
        rounded,
        packed0,
        packed1,
        gates,
    })
}

fn check_paired_packed_case(count: usize, width: usize) -> crate::Result<()> {
    let offset = 2;
    let fixture = build_paired_fixture(count, width)?;
    let borrowed_gates = fixture
        .gates
        .each_ref()
        .map(|row| row.each_ref().map(Vec::as_slice));
    let borrowed_scores: [[&[f32]; 6]; 2] = std::array::from_fn(|token| {
        std::array::from_fn(|head| fixture.normalized[token][head].as_slice())
    });
    let borrowed_rounded: [[&[f32]; 6]; 2] = std::array::from_fn(|token| {
        std::array::from_fn(|head| fixture.rounded[token][head].as_slice())
    });

    let mut expected: [[Vec<f32>; 6]; 2] =
        std::array::from_fn(|_| std::array::from_fn(|_| vec![19.0; width]));
    value_pair_tile_gated(
        fixture.rows.iter().map(Vec::as_slice),
        fixture.current.each_ref().map(Vec::as_slice),
        borrowed_rounded,
        offset,
        outputs(&mut expected),
        borrowed_gates,
    )?;

    let visits_packed = Cell::new(0);
    let mut actual_packed: [[Vec<f32>; 6]; 2] =
        std::array::from_fn(|_| std::array::from_fn(|_| vec![19.0; width]));
    value_pair_tile_packed_gated(
        fixture.rows.iter().map(|row| {
            visits_packed.set(visits_packed.get() + 1);
            row.as_slice()
        }),
        fixture.current.each_ref().map(Vec::as_slice),
        [&fixture.packed0, &fixture.packed1],
        offset,
        outputs(&mut actual_packed),
        borrowed_gates,
    )?;
    assert_eq!(visits_packed.get(), count);

    let visits_scores = Cell::new(0);
    let mut actual_scores: [[Vec<f32>; 6]; 2] =
        std::array::from_fn(|_| std::array::from_fn(|_| vec![19.0; width]));
    value_pair_tile_from_scores_gated(
        fixture.rows.iter().map(|row| {
            visits_scores.set(visits_scores.get() + 1);
            row.as_slice()
        }),
        fixture.current.each_ref().map(Vec::as_slice),
        borrowed_scores,
        offset,
        outputs(&mut actual_scores),
        borrowed_gates,
    )?;
    assert_eq!(visits_scores.get(), count);

    let mut scalar_packed_out = actual_packed.clone();
    scalar_packed(
        fixture.rows.iter().map(Vec::as_slice),
        fixture.current.each_ref().map(Vec::as_slice),
        [&fixture.packed0, &fixture.packed1],
        &(offset..offset + width),
        outputs(&mut scalar_packed_out),
        borrowed_gates,
    )?;
    let mut scalar_scores_out = actual_scores.clone();
    scalar_from_scores(
        fixture.rows.iter().map(Vec::as_slice),
        fixture.current.each_ref().map(Vec::as_slice),
        borrowed_scores,
        &(offset..offset + width),
        outputs(&mut scalar_scores_out),
        borrowed_gates,
    )?;

    for out in [
        &actual_packed,
        &actual_scores,
        &scalar_packed_out,
        &scalar_scores_out,
    ] {
        assert_pair_outputs_eq(out, &expected);
    }
    Ok(())
}

#[test]
fn paired_packed_and_chunked_score_values_match_f32_oracle_across_chunks() -> crate::Result<()> {
    for count in [1, 2, 15, 16, 17, 64, 65, 255, 256, 257, 273] {
        for width in [1, 3, 4, 8, 31, 32] {
            check_paired_packed_case(count, width)?;
        }
    }
    Ok(())
}

#[test]
fn paired_packed_and_chunked_score_values_reject_malformed_before_publication() {
    let packed0 = [[0x3c00_u16; 6]; 16];
    let packed1 = [[0x3c00_u16; 6]; 17];
    let scores0 = [1.0_f32; 16];
    let scores1 = [1.0_f32; 17];
    let gates = [[&[0.0; 4][..]; 6]; 2];
    let rows = [[0x3c00_u16; 4]; 17];
    let mut actual: [[Vec<f32>; 6]; 2] =
        std::array::from_fn(|_| std::array::from_fn(|_| vec![19.0; 4]));
    let retained = actual.clone();

    for count in [0, 15, 17] {
        assert!(
            value_pair_tile_packed_gated(
                rows[..count].iter().map(<[u16; 4]>::as_slice),
                [&[1.0; 4]; 2],
                [&packed0, &packed1],
                0,
                outputs(&mut actual),
                gates,
            )
            .is_err()
        );
        assert_eq!(actual, retained);
        assert!(
            value_pair_tile_from_scores_gated(
                rows[..count].iter().map(<[u16; 4]>::as_slice),
                [&[1.0; 4]; 2],
                [[&scores0[..]; 6], [&scores1[..]; 6]],
                0,
                outputs(&mut actual),
                gates,
            )
            .is_err()
        );
        assert_eq!(actual, retained);
    }

    let mut bad_packed = packed1;
    bad_packed[16][5] = 0x7c00;
    assert!(
        value_pair_tile_packed_gated(
            rows[..16].iter().map(<[u16; 4]>::as_slice),
            [&[1.0; 4]; 2],
            [&packed0, &bad_packed],
            0,
            outputs(&mut actual),
            gates,
        )
        .is_err()
    );
    assert_eq!(actual, retained);

    let mut overflow_scores = scores1;
    overflow_scores[16] = 65_520.0;
    let mut bad_scores = [[&scores0[..]; 6], [&scores1[..]; 6]];
    bad_scores[1][5] = &overflow_scores;
    assert!(
        value_pair_tile_from_scores_gated(
            rows[..16].iter().map(<[u16; 4]>::as_slice),
            [&[1.0; 4]; 2],
            bad_scores,
            0,
            outputs(&mut actual),
            gates,
        )
        .is_err()
    );
    assert_eq!(actual, retained);
}
