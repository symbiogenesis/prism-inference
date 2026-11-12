//! Direct controls for private GDN staging, independent of SIMD dispatch fallback.

use super::super::{ChunkInput, dot_scalar, scalar_add_scaled};
use crate::{Result, invalid};

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

fn values(count: usize) -> Vec<f32> {
    let mut seed = 0x9e37_79b9_u32;
    (0..count)
        .map(|_| {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            f32::from_bits((seed & 0x807f_ffff) | 0x3d00_0000)
        })
        .collect()
}

fn mutable_rows<const ROWS: usize>(state: &mut [f32], width: usize) -> Result<[&mut [f32]; ROWS]> {
    state
        .chunks_exact_mut(width)
        .collect::<Vec<_>>()
        .try_into()
        .map_err(|_| invalid("invalid test row geometry"))
}

fn oracle_row(
    state: &mut [f32],
    inputs: &[ChunkInput<'_>],
    row: usize,
    stride: usize,
    output: &mut [f32],
) -> Result<()> {
    for (token, input) in inputs.iter().enumerate() {
        for element in &mut *state {
            *element *= input.decay;
        }
        let dot = dot_scalar(state, input.key)?;
        let delta = (input.value[row] - dot) * input.beta;
        scalar_add_scaled(state, input.key, delta)?;
        output[token * stride + row] = dot_scalar(state, input.query)? * 0.125;
    }
    Ok(())
}

fn decode_oracle(
    state: &mut [f32],
    key: &[f32],
    query: &[f32],
    decay: f32,
    value: f32,
    beta: f32,
) -> Result<f32> {
    for element in &mut *state {
        *element *= decay;
    }
    let dot = dot_scalar(state, key)?;
    let delta = (value - dot) * beta;
    scalar_add_scaled(state, key, delta)?;
    dot_scalar(state, query)
}

#[test]
fn pinned_gdn_decode_capacities_match_independent_scalar_bits() -> Result<()> {
    for width in [16, 112, 128, 144, 240, 256] {
        let mut previous = values(width);
        previous[0] = -0.0;
        previous[1] = f32::from_bits(1);
        previous[2] = -f32::MIN_POSITIVE;
        let key = values(width);
        let mut query = values(width);
        query.reverse();
        for (decay, value, beta) in [
            (0.875, 0.125, 0.625),
            (0.0, -0.0, 0.0),
            (-0.0, 0.125, -0.0),
            (-0.25, -0.25, 1.0),
        ] {
            let mut expected = previous.clone();
            let expected_output = decode_oracle(&mut expected, &key, &query, decay, value, beta)?;
            let mut actual = vec![-5.0; width];
            let actual_output =
                super::step_row_from(&previous, &mut actual, &key, &query, decay, value, beta)
                    .ok_or_else(|| invalid("baseline NEON decode was not selected"))??;
            assert_eq!(bits(&actual), bits(&expected));
            assert_eq!(actual_output.to_bits(), expected_output.to_bits());
            let mut in_place = previous.clone();
            let in_place_output =
                super::step_row_in_place(&mut in_place, &key, &query, decay, value, beta)
                    .ok_or_else(|| invalid("baseline NEON decode was not selected"))??;
            assert_eq!(bits(&in_place), bits(&expected));
            assert_eq!(in_place_output.to_bits(), expected_output.to_bits());
        }
    }
    Ok(())
}

#[test]
fn pinned_gdn_decode_capacities_preserve_failure_and_refusal_bits() -> Result<()> {
    for width in [16, 128, 144, 256] {
        for failure in 0..3 {
            let previous = values(width);
            let mut key = values(width);
            let mut query = values(width);
            let value = if failure == 1 { f32::MAX } else { 0.125 };
            match failure {
                0 => key[width - 1] = f32::NAN,
                1 => key[width / 2] = 2.0,
                _ => query[width - 1] = f32::NAN,
            }
            let mut expected = previous.clone();
            assert!(decode_oracle(&mut expected, &key, &query, 0.875, value, 1.0).is_err());
            let mut actual = vec![-5.0; width];
            let result =
                super::step_row_from(&previous, &mut actual, &key, &query, 0.875, value, 1.0)
                    .ok_or_else(|| invalid("baseline NEON decode was not selected"))?;
            assert!(result.is_err());
            assert_eq!(bits(&actual), bits(&expected));
            let mut in_place = previous;
            let result = super::step_row_in_place(&mut in_place, &key, &query, 0.875, value, 1.0)
                .ok_or_else(|| invalid("baseline NEON decode was not selected"))?;
            assert!(result.is_err());
            assert_eq!(bits(&in_place), bits(&expected));
        }
    }
    for width in [0, 1, 15, 17, 127, 129, 255, 257, 272] {
        let previous = values(width);
        let key = values(width);
        let query = values(width);
        let mut actual = vec![-5.0; width];
        let original = bits(&actual);
        assert!(
            super::step_row_from(&previous, &mut actual, &key, &query, 0.875, 0.125, 0.625)
                .is_none()
        );
        assert_eq!(bits(&actual), original);
        assert!(super::step_row_in_place(&mut actual, &key, &query, 0.875, 0.125, 0.625).is_none());
        assert_eq!(bits(&actual), original);
    }
    Ok(())
}

fn grouped_decode(
    previous: &[f32],
    state: &mut [f32],
    rows: usize,
    width: usize,
    key: &[f32],
    query: &[f32],
    token_values: &[f32],
) -> Result<Option<Vec<f32>>> {
    Ok(if rows == 4 {
        super::step_four_rows_from(
            std::array::from_fn(|row| &previous[row * width..(row + 1) * width]),
            mutable_rows(state, width)?,
            key,
            query,
            0.875,
            std::array::from_fn(|row| token_values[row]),
            0.625,
        )
        .map(Vec::from)
    } else {
        super::step_eight_rows_from(
            std::array::from_fn(|row| &previous[row * width..(row + 1) * width]),
            mutable_rows(state, width)?,
            key,
            query,
            0.875,
            std::array::from_fn(|row| token_values[row]),
            0.625,
        )
        .map(Vec::from)
    })
}

#[test]
fn pinned_gdn_grouped_decode_capacities_match_independent_scalar_bits() -> Result<()> {
    for width in [16, 128, 144, 256] {
        let key = values(width);
        let mut query = values(width);
        query.reverse();
        for rows in [4, 8] {
            let mut previous = values(width * rows);
            previous[0] = -0.0;
            previous[1] = f32::from_bits(1);
            let token_values = values(rows);
            let mut expected = previous.clone();
            let expected_output = expected
                .chunks_exact_mut(width)
                .zip(&token_values)
                .map(|(state, &value)| decode_oracle(state, &key, &query, 0.875, value, 0.625))
                .collect::<Result<Vec<_>>>()?;
            let mut actual = vec![-5.0; width * rows];
            let actual_output = grouped_decode(
                &previous,
                &mut actual,
                rows,
                width,
                &key,
                &query,
                &token_values,
            )?
            .ok_or_else(|| invalid("baseline NEON grouped decode was not selected"))?;
            assert_eq!(bits(&actual), bits(&expected));
            assert_eq!(bits(&actual_output), bits(&expected_output));
        }
    }
    Ok(())
}

#[test]
fn pinned_gdn_grouped_decode_capacities_keep_failure_atomic() -> Result<()> {
    for width in [128, 144] {
        for rows in [4, 8] {
            for failure in 0..3 {
                let previous = values(width * rows);
                let mut key = values(width);
                let mut query = values(width);
                let mut token_values = values(rows);
                match failure {
                    0 => key[width - 1] = f32::NAN,
                    1 => {
                        token_values[rows - 1] = f32::MAX;
                        key[width / 2] = 2.0;
                    }
                    _ => query[width - 1] = f32::NAN,
                }
                let mut actual = vec![-5.0; width * rows];
                let original = bits(&actual);
                assert!(
                    grouped_decode(
                        &previous,
                        &mut actual,
                        rows,
                        width,
                        &key,
                        &query,
                        &token_values,
                    )?
                    .is_none()
                );
                assert_eq!(bits(&actual), original);
            }
        }
    }
    for width in [129, 272] {
        for rows in [4, 8] {
            let previous = values(width * rows);
            let key = values(width);
            let query = values(width);
            let token_values = values(rows);
            let mut actual = vec![-5.0; width * rows];
            let original = bits(&actual);
            assert!(
                grouped_decode(
                    &previous,
                    &mut actual,
                    rows,
                    width,
                    &key,
                    &query,
                    &token_values,
                )?
                .is_none()
            );
            assert_eq!(bits(&actual), original);
        }
    }
    Ok(())
}

fn grouped_from(
    previous: &[f32],
    state: &mut [f32],
    rows: usize,
    width: usize,
    inputs: &[ChunkInput<'_>],
    output: &mut [f32],
) -> Result<bool> {
    Ok(if rows == 4 {
        super::step_four_rows_tokens_from(
            std::array::from_fn(|row| &previous[row * width..(row + 1) * width]),
            mutable_rows(state, width)?,
            inputs,
            0,
            width,
            0.125,
            output,
        )
    } else {
        super::step_eight_rows_tokens_from(
            std::array::from_fn(|row| &previous[row * width..(row + 1) * width]),
            mutable_rows(state, width)?,
            inputs,
            0,
            width,
            0.125,
            output,
        )
    })
}

fn grouped_in_place(
    state: &mut [f32],
    rows: usize,
    width: usize,
    inputs: &[ChunkInput<'_>],
) -> Result<bool> {
    Ok(if rows == 4 {
        super::step_four_rows_tokens_in_place(mutable_rows(state, width)?, inputs, 0)
    } else {
        super::step_eight_rows_tokens_in_place(mutable_rows(state, width)?, inputs, 0)
    })
}

#[test]
fn pinned_gdn_grouped_finite_masks_keep_late_failures_atomic() -> Result<()> {
    for width in [128, 144, 256] {
        for rows in [4, 8] {
            for failure_row in [0, rows - 1] {
                for failure_lane in [0, width - 1] {
                    for failure in 0..4 {
                        let mut previous = vec![0.25; width * rows];
                        let key = vec![0.0; width];
                        let query = vec![0.125; width];
                        let token_values = vec![0.125; width];
                        let mut failing_key = key.clone();
                        let mut failing_query = query.clone();
                        let mut failing_values = token_values.clone();
                        match failure {
                            0 => failing_key[failure_lane] = f32::NAN,
                            1 => {
                                // Finite decay and key dot; the update overflows
                                // only in the selected row and vector lane.
                                failing_key[failure_lane] = 4.0;
                                failing_values[failure_row] = f32::MAX;
                            }
                            2 => {
                                // Decay remains finite; the key dot overflows.
                                previous[failure_row * width + failure_lane] = f32::MAX;
                                failing_key[failure_lane] = 4.0;
                            }
                            _ => failing_query.fill(f32::MAX),
                        }
                        let mut decode_state = vec![-5.0; width * rows];
                        let untouched = bits(&decode_state);
                        assert!(
                            grouped_decode(
                                &previous,
                                &mut decode_state,
                                rows,
                                width,
                                &failing_key,
                                &failing_query,
                                &failing_values,
                            )?
                            .is_none()
                        );
                        assert_eq!(bits(&decode_state), untouched);

                        for count in [1, 8, 32] {
                            let ordinary = ChunkInput {
                                query: &query,
                                key: &key,
                                value: &token_values,
                                decay: 1.0,
                                beta: 0.625,
                            };
                            let mut inputs = vec![ordinary; count];
                            inputs[count - 1] = ChunkInput {
                                query: &failing_query,
                                key: &failing_key,
                                value: &failing_values,
                                ..ordinary
                            };
                            let mut state = vec![-5.0; width * rows];
                            let mut output = vec![-3.0; width * count];
                            let original_output = bits(&output);
                            assert!(!grouped_from(
                                &previous,
                                &mut state,
                                rows,
                                width,
                                &inputs,
                                &mut output,
                            )?);
                            assert_eq!(bits(&state), untouched);
                            assert_eq!(bits(&output), original_output);
                            let mut in_place = previous.clone();
                            assert!(!grouped_in_place(&mut in_place, rows, width, &inputs)?);
                            assert_eq!(bits(&in_place), bits(&previous));
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

#[test]
fn pinned_gdn_grouped_finite_masks_replay_original_row_failure_bits() -> Result<()> {
    for width in [128, 144] {
        for rows in [4, 8] {
            for failure_row in [0, rows - 1] {
                let previous = values(width * rows);
                let query = values(width);
                let key = values(width);
                let token_values = values(width);
                let mut failing_key = key.clone();
                failing_key[width - 1] = 4.0;
                let mut failing_values = token_values.clone();
                failing_values[failure_row] = f32::MAX;
                let ordinary = ChunkInput {
                    query: &query,
                    key: &key,
                    value: &token_values,
                    decay: 0.875,
                    beta: 0.625,
                };
                let inputs = [
                    ordinary,
                    ordinary,
                    ChunkInput {
                        key: &failing_key,
                        value: &failing_values,
                        ..ordinary
                    },
                ];
                let mut expected = vec![-5.0; width * rows];
                let mut expected_output = vec![-3.0; width * inputs.len()];
                let mut failed = false;
                for row in 0..rows {
                    let range = row * width..(row + 1) * width;
                    expected[range.clone()].copy_from_slice(&previous[range.clone()]);
                    if oracle_row(
                        &mut expected[range],
                        &inputs,
                        row,
                        width,
                        &mut expected_output,
                    )
                    .is_err()
                    {
                        failed = true;
                        break;
                    }
                }
                assert!(failed);
                let mut actual = vec![-5.0; width * rows];
                let mut actual_output = vec![-3.0; width * inputs.len()];
                let result = if rows == 4 {
                    super::super::step_four_rows_tokens_from(
                        std::array::from_fn(|row| &previous[row * width..(row + 1) * width]),
                        mutable_rows(&mut actual, width)?,
                        &inputs,
                        0,
                        width,
                        0.125,
                        &mut actual_output,
                    )
                } else {
                    super::super::step_eight_rows_tokens_from(
                        std::array::from_fn(|row| &previous[row * width..(row + 1) * width]),
                        mutable_rows(&mut actual, width)?,
                        &inputs,
                        0,
                        width,
                        0.125,
                        &mut actual_output,
                    )
                };
                assert!(result.is_err());
                assert_eq!(bits(&actual), bits(&expected));
                assert_eq!(bits(&actual_output), bits(&expected_output));
            }
        }
    }
    Ok(())
}

#[test]
fn pinned_gdn_staged_rows_match_independent_scalar_bits() -> Result<()> {
    for width in [16, 64, 112, 128, 144, 240, 256] {
        let query = values(width);
        let mut key = values(width);
        key.reverse();
        let mut token_values = values(width);
        token_values[0] = -0.0;
        for count in [0, 1, 3, 8, 16, 32, 64] {
            let inputs: Vec<ChunkInput<'_>> = (0..count)
                .map(|token| ChunkInput {
                    query: &query,
                    key: &key,
                    value: &token_values,
                    decay: [0.875, -0.0, 0.5, -0.25][token % 4],
                    beta: [0.625, 0.0, -0.0, 0.375][token % 4],
                })
                .collect();
            for rows in [4, 8] {
                let previous = values(width * rows);
                let mut expected = previous.clone();
                let mut expected_output = vec![-3.0; width * count];
                for (row, state) in expected.chunks_exact_mut(width).enumerate() {
                    oracle_row(state, &inputs, row, width, &mut expected_output)?;
                }
                let mut actual = vec![-5.0; width * rows];
                let mut actual_output = vec![-3.0; width * count];
                assert!(grouped_from(
                    &previous,
                    &mut actual,
                    rows,
                    width,
                    &inputs,
                    &mut actual_output,
                )?);
                assert_eq!(bits(&actual), bits(&expected));
                assert_eq!(bits(&actual_output), bits(&expected_output));
                let mut in_place = previous;
                assert!(grouped_in_place(&mut in_place, rows, width, &inputs)?);
                assert_eq!(bits(&in_place), bits(&expected));
            }
        }
    }
    Ok(())
}

#[test]
fn pinned_gdn_staged_group_failure_keeps_destinations_untouched() -> Result<()> {
    for width in [128, 144] {
        let query = values(width);
        let key = values(width);
        let token_values = values(width);
        for rows in [4, 8] {
            for failure in 0..3 {
                let mut failing_query = query.clone();
                let mut failing_key = key.clone();
                let mut failing_values = token_values.clone();
                match failure {
                    0 => failing_key[width - 1] = f32::NAN,
                    1 => {
                        failing_values[rows - 1] = f32::MAX;
                        failing_key[21] = 2.0;
                    }
                    _ => failing_query[width - 1] = f32::NAN,
                }
                let ordinary = ChunkInput {
                    query: &query,
                    key: &key,
                    value: &token_values,
                    decay: 0.875,
                    beta: 0.625,
                };
                let inputs = [
                    ordinary,
                    ordinary,
                    ChunkInput {
                        query: &failing_query,
                        key: &failing_key,
                        value: &failing_values,
                        beta: 1.0,
                        ..ordinary
                    },
                ];
                let previous = values(width * rows);
                let mut state = vec![-5.0; width * rows];
                let original_state = bits(&state);
                let mut output = vec![-3.0; width * inputs.len()];
                let original_output = bits(&output);
                assert!(!grouped_from(
                    &previous,
                    &mut state,
                    rows,
                    width,
                    &inputs,
                    &mut output,
                )?);
                assert_eq!(bits(&state), original_state);
                assert_eq!(bits(&output), original_output);
                let mut in_place = previous.clone();
                assert!(!grouped_in_place(&mut in_place, rows, width, &inputs)?);
                assert_eq!(bits(&in_place), bits(&previous));
            }
        }
    }
    Ok(())
}

#[test]
fn pinned_gdn_staged_single_row_retains_late_failure_bits() -> Result<()> {
    for width in [16, 128, 144, 256] {
        for failure in 0..3 {
            let previous = values(width);
            let query = values(width);
            let key = values(width);
            let token_values = values(width);
            let mut failing_query = query.clone();
            let mut failing_key = key.clone();
            let mut failing_values = token_values.clone();
            match failure {
                0 => failing_key[width - 1] = f32::NAN,
                1 => {
                    failing_values[0] = f32::MAX;
                    failing_key[width / 2] = 2.0;
                }
                _ => failing_query[width - 1] = f32::NAN,
            }
            let ordinary = ChunkInput {
                query: &query,
                key: &key,
                value: &token_values,
                decay: 0.875,
                beta: 0.625,
            };
            let inputs = [
                ordinary,
                ordinary,
                ChunkInput {
                    query: &failing_query,
                    key: &failing_key,
                    value: &failing_values,
                    beta: 1.0,
                    ..ordinary
                },
            ];
            let mut expected = previous.clone();
            let mut expected_output = vec![-3.0; width * inputs.len()];
            assert!(oracle_row(&mut expected, &inputs, 0, width, &mut expected_output).is_err());
            let mut actual = vec![-5.0; width];
            let mut actual_output = vec![-3.0; width * inputs.len()];
            let result = super::step_row_tokens_from(
                &previous,
                &mut actual,
                &inputs,
                0,
                width,
                0.125,
                &mut actual_output,
            )
            .ok_or_else(|| invalid("baseline NEON control was not selected"))?;
            assert!(result.is_err());
            assert_eq!(bits(&actual), bits(&expected));
            assert_eq!(bits(&actual_output), bits(&expected_output));
        }
    }
    Ok(())
}

#[test]
fn pinned_gdn_multitoken_single_row_capacities_match_independent_scalar_bits() -> Result<()> {
    for width in [16, 112, 128, 144, 240, 256] {
        let key = values(width);
        let mut query = values(width);
        query.reverse();
        let token_values = [-0.0];
        for count in [0, 1, 3, 8, 16, 32, 64] {
            let inputs: Vec<ChunkInput<'_>> = (0..count)
                .map(|token| ChunkInput {
                    query: &query,
                    key: &key,
                    value: &token_values,
                    decay: [0.875, -0.0, 0.5, -0.25][token % 4],
                    beta: [0.625, 0.0, -0.0, 0.375][token % 4],
                })
                .collect();
            let mut previous = values(width);
            previous[0] = -0.0;
            previous[1] = f32::from_bits(1);
            let mut expected = previous.clone();
            let mut expected_output = vec![-3.0; count * 3];
            oracle_row(&mut expected, &inputs, 0, 3, &mut expected_output)?;
            let mut actual = vec![-5.0; width];
            let mut actual_output = vec![-3.0; count * 3];
            super::step_row_tokens_from(
                &previous,
                &mut actual,
                &inputs,
                0,
                3,
                0.125,
                &mut actual_output,
            )
            .ok_or_else(|| invalid("baseline NEON multi-token control was not selected"))??;
            assert_eq!(bits(&actual), bits(&expected));
            assert_eq!(bits(&actual_output), bits(&expected_output));
        }
    }
    Ok(())
}

#[test]
fn pinned_gdn_multitoken_capacities_preserve_refusal() -> Result<()> {
    for width in [1, 15, 17, 127, 129, 255, 257, 272] {
        let previous = values(width);
        let mut actual = vec![-5.0; width];
        let before = bits(&actual);
        let mut output = [-3.0; 3];
        assert!(
            super::step_row_tokens_from(&previous, &mut actual, &[], 0, 1, 0.125, &mut output,)
                .is_none()
        );
        assert_eq!(bits(&actual), before);
        assert_eq!(output, [-3.0; 3]);
        for rows in [4, 8] {
            let previous = values(width * rows);
            let mut actual = vec![-5.0; width * rows];
            let before = bits(&actual);
            assert!(!grouped_from(
                &previous,
                &mut actual,
                rows,
                width,
                &[],
                &mut output,
            )?);
            assert_eq!(bits(&actual), before);
            assert!(!grouped_in_place(&mut actual, rows, width, &[])?);
            assert_eq!(bits(&actual), before);
            assert_eq!(output, [-3.0; 3]);
        }
    }
    for width in [128, 144] {
        let key = values(width);
        let query = values(width);
        let token_values = values(width);
        let inputs = [ChunkInput {
            query: &query,
            key: &key,
            value: &token_values,
            decay: 0.875,
            beta: 0.625,
        }; 65];
        for rows in [4, 8] {
            let previous = values(width * rows);
            let mut actual = vec![-5.0; width * rows];
            let before = bits(&actual);
            let mut output = vec![-3.0; 65 * width];
            let before_output = bits(&output);
            assert!(!grouped_from(
                &previous,
                &mut actual,
                rows,
                width,
                &inputs,
                &mut output,
            )?);
            assert_eq!(bits(&actual), before);
            assert!(!grouped_in_place(&mut actual, rows, width, &inputs)?);
            assert_eq!(bits(&actual), before);
            assert_eq!(bits(&output), before_output);
        }
    }
    Ok(())
}

#[test]
fn x1_gdn_four_row_phases_keep_eight_row_temporal_bits() -> Result<()> {
    for width in [16, 112, 128, 144, 256] {
        let mut actual = values(width * 8);
        actual[0] = -0.0;
        actual[4 * width] = f32::from_bits(1);
        actual[7 * width] = -f32::MIN_POSITIVE;
        let mut expected = actual.clone();
        let mut key = values(width);
        let mut query = values(width);
        query.reverse();
        for token in 0..8 {
            key.rotate_left(1);
            query.rotate_right(1);
            let decay = [0.875, -0.0, -0.25, 1.0][token % 4];
            let beta = [0.625, 0.0, -0.0, 0.375][token % 4];
            let token_values: [f32; 8] =
                std::array::from_fn(|row| if row == token { -0.0 } else { key[row % width] });
            let expected_output = expected
                .chunks_exact_mut(width)
                .zip(token_values)
                .map(|(state, value)| decode_oracle(state, &key, &query, decay, value, beta))
                .collect::<Result<Vec<_>>>()?;
            let previous = actual;
            actual = vec![-5.0; width * 8];
            let output = super::step_eight_rows_from(
                std::array::from_fn(|row| &previous[row * width..(row + 1) * width]),
                mutable_rows(&mut actual, width)?,
                &key,
                &query,
                decay,
                token_values,
                beta,
            )
            .ok_or_else(|| invalid("four-row GDN arithmetic phases were not selected"))?;
            assert_eq!(bits(&actual), bits(&expected));
            assert_eq!(bits(&output), bits(&expected_output));
        }
    }
    Ok(())
}

#[test]
fn x1_gdn_second_four_row_phase_keeps_atomic_refusal_and_scalar_failure_bits() -> Result<()> {
    for width in [16, 128, 144, 256] {
        for failing_row in [4, 7] {
            for failure in 0..3 {
                let mut previous = vec![0.0; width * 8];
                let mut key = vec![0.0; width];
                let mut query = vec![0.0; width];
                let mut token_values = [0.0; 8];
                match failure {
                    0 => {
                        previous[(failing_row + 1) * width - 1] = f32::MAX;
                        key[width - 1] = 4.0;
                    }
                    1 => {
                        token_values[failing_row] = f32::MAX;
                        key[width - 1] = 4.0;
                    }
                    _ => {
                        token_values[failing_row] = 1.0;
                        key.fill(1.0);
                        query.fill(f32::MAX);
                    }
                }
                let previous_rows =
                    std::array::from_fn(|row| &previous[row * width..(row + 1) * width]);
                let mut state = vec![-5.0; width * 8];
                let untouched = bits(&state);
                assert!(
                    super::step_eight_rows_from(
                        previous_rows,
                        mutable_rows(&mut state, width)?,
                        &key,
                        &query,
                        1.0,
                        token_values,
                        1.0,
                    )
                    .is_none()
                );
                assert_eq!(bits(&state), untouched);

                // The public parent replays every row from immutable previous
                // state after refusal. Compare the exact finite prefix, failing
                // row and untouched later rows with independent scalar steps.
                let mut expected = state.clone();
                let mut failed = false;
                for (row, target) in expected.chunks_exact_mut(width).enumerate() {
                    target.copy_from_slice(previous_rows[row]);
                    if decode_oracle(target, &key, &query, 1.0, token_values[row], 1.0).is_err() {
                        failed = true;
                        break;
                    }
                }
                assert!(failed);
                assert!(
                    super::super::step_eight_rows_from(
                        previous_rows,
                        mutable_rows(&mut state, width)?,
                        &key,
                        &query,
                        1.0,
                        token_values,
                        1.0,
                    )
                    .is_err()
                );
                assert_eq!(bits(&state), bits(&expected));
            }
        }
    }
    Ok(())
}

#[test]
fn x1_gdn_prefill_four_row_phases_match_scalar_and_repeated_decode_bits() -> Result<()> {
    for width in [16, 112, 128, 144, 256] {
        for count in [1, 2, 7, 32, 64] {
            let keys: Vec<_> = (0..count)
                .map(|token| {
                    let mut key = values(width);
                    key.rotate_left(token % width);
                    key
                })
                .collect();
            let queries: Vec<_> = (0..count)
                .map(|token| {
                    let mut query = values(width);
                    query.rotate_right((token + 1) % width);
                    query
                })
                .collect();
            let token_values: Vec<_> = (0..count)
                .map(|token| {
                    let mut value = values(8);
                    value[token % 8] = -0.0;
                    value
                })
                .collect();
            let inputs: Vec<_> = (0..count)
                .map(|token| ChunkInput {
                    key: &keys[token],
                    query: &queries[token],
                    value: &token_values[token],
                    decay: [0.875, -0.0, -0.25, 1.0][token % 4],
                    beta: [0.625, 0.0, -0.0, 0.375][token % 4],
                })
                .collect();
            let mut previous = values(width * 8);
            previous[0] = -0.0;
            previous[4 * width] = f32::from_bits(1);
            previous[8 * width - 1] = -f32::MIN_POSITIVE;
            let mut expected = previous.clone();
            let mut expected_output = vec![-3.0; count * width];
            for (row, state) in expected.chunks_exact_mut(width).enumerate() {
                oracle_row(state, &inputs, row, width, &mut expected_output)?;
            }
            let mut repeated = previous.clone();
            for (token, input) in inputs.iter().enumerate() {
                let before = repeated;
                repeated = vec![-5.0; width * 8];
                let output = super::step_eight_rows_from(
                    std::array::from_fn(|row| &before[row * width..(row + 1) * width]),
                    mutable_rows(&mut repeated, width)?,
                    input.key,
                    input.query,
                    input.decay,
                    std::array::from_fn(|row| input.value[row]),
                    input.beta,
                )
                .ok_or_else(|| invalid("eight-row decode control refused"))?;
                let expected = &expected_output[token * width..token * width + 8];
                assert_eq!(bits(&output.map(|value| value * 0.125)), bits(expected));
            }
            assert_eq!(bits(&repeated), bits(&expected));
            let mut actual = vec![-5.0; width * 8];
            let mut output = vec![-3.0; count * width];
            assert!(grouped_from(
                &previous,
                &mut actual,
                8,
                width,
                &inputs,
                &mut output
            )?);
            assert_eq!(bits(&actual), bits(&expected));
            assert_eq!(bits(&output), bits(&expected_output));
            let mut in_place = previous;
            assert!(grouped_in_place(&mut in_place, 8, width, &inputs)?);
            assert_eq!(bits(&in_place), bits(&expected));
        }
    }
    Ok(())
}

#[test]
fn x1_gdn_prefill_second_phase_late_failure_keeps_all_destinations_untouched() -> Result<()> {
    for width in [16, 128, 144, 256] {
        for failing_row in [4, 7] {
            for failing_token in [0, 2, 6] {
                for failure in 0..3 {
                    let count = 7;
                    let mut previous = vec![0.0; width * 8];
                    let mut keys = vec![vec![0.0; width]; count];
                    let mut queries = vec![vec![0.0; width]; count];
                    let mut token_values = vec![[0.0; 8]; count];
                    match failure {
                        0 => {
                            previous[(failing_row + 1) * width - 1] = f32::MAX;
                            keys[failing_token][width - 1] = 4.0;
                        }
                        1 => {
                            token_values[failing_token][failing_row] = f32::MAX;
                            keys[failing_token][width - 1] = 4.0;
                        }
                        _ => {
                            token_values[failing_token][failing_row] = 1.0;
                            keys[failing_token].fill(1.0);
                            queries[failing_token].fill(f32::MAX);
                        }
                    }
                    let inputs: Vec<_> = (0..count)
                        .map(|token| ChunkInput {
                            key: &keys[token],
                            query: &queries[token],
                            value: &token_values[token],
                            decay: 1.0,
                            beta: if token == failing_token { 1.0 } else { 0.0 },
                        })
                        .collect();
                    let mut oracle =
                        previous[failing_row * width..(failing_row + 1) * width].to_vec();
                    assert!(
                        oracle_row(
                            &mut oracle,
                            &inputs,
                            failing_row,
                            width,
                            &mut vec![-3.0; count * width]
                        )
                        .is_err()
                    );
                    let mut actual = vec![-5.0; width * 8];
                    let untouched = bits(&actual);
                    let mut output = vec![-3.0; count * width];
                    let untouched_output = bits(&output);
                    assert!(!grouped_from(
                        &previous,
                        &mut actual,
                        8,
                        width,
                        &inputs,
                        &mut output
                    )?);
                    assert_eq!(bits(&actual), untouched);
                    assert_eq!(bits(&output), untouched_output);
                    let mut in_place = previous.clone();
                    assert!(!grouped_in_place(&mut in_place, 8, width, &inputs)?);
                    assert_eq!(bits(&in_place), bits(&previous));
                }
            }
        }
    }
    Ok(())
}
