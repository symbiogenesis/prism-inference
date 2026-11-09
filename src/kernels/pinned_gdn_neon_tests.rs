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
fn pinned_gdn_staged_rows_match_independent_scalar_bits() -> Result<()> {
    for width in [16, 64, 128, 256] {
        let query = values(width);
        let mut key = values(width);
        key.reverse();
        let mut token_values = values(width);
        token_values[0] = -0.0;
        for count in [1, 3, 8, 32] {
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
    let width = 128;
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
    Ok(())
}

#[test]
fn pinned_gdn_staged_single_row_retains_late_failure_bits() -> Result<()> {
    for width in [16, 128, 256] {
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
