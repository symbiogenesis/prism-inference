//! Disposable GDN decode state, independent arithmetic and retry controls.

use super::{
    ChunkInput, dot_scalar, scalar_add_scaled, step_eight_rows_from, step_eight_rows_provisional,
    try_step_head_provisional,
};
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

fn rows<const ROWS: usize>(state: &mut [f32], width: usize) -> Result<[&mut [f32]; ROWS]> {
    state
        .chunks_exact_mut(width)
        .collect::<Vec<_>>()
        .try_into()
        .map_err(|_| invalid("invalid test row geometry"))
}

fn oracle(
    previous: &[f32],
    key: &[f32],
    query: &[f32],
    decay: f32,
    input: [f32; 8],
    beta: f32,
) -> Result<(Vec<f32>, [f32; 8])> {
    let mut state = previous.to_vec();
    let mut outputs = [0.0; 8];
    for ((state, value), output) in state
        .chunks_exact_mut(key.len())
        .zip(input)
        .zip(&mut outputs)
    {
        for element in &mut *state {
            *element *= decay;
        }
        let delta = (value - dot_scalar(state, key)?) * beta;
        scalar_add_scaled(state, key, delta)?;
        *output = dot_scalar(state, query)?;
    }
    Ok((state, outputs))
}

#[test]
fn x1_provisional_gdn_decode_matches_scalar_bits_across_reused_deltas() -> Result<()> {
    for width in [1, 15, 16, 112, 128, 144, 240, 256, 257] {
        let mut committed = values(8 * width);
        committed[0] = -0.0;
        committed[1] = f32::from_bits(1);
        committed[2] = -f32::MIN_POSITIVE;
        let key = values(width);
        let mut query = values(width);
        query.reverse();
        let mut destination = vec![f32::NAN; committed.len()];
        for (decay, value, beta) in [
            (0.875, 0.125, 0.625),
            (0.0, -0.0, 0.0),
            (-0.0, 0.125, -0.0),
            (-0.25, -0.25, 1.0),
        ] {
            let input = [value; 8];
            let before = bits(&committed);
            let (expected, expected_output) = oracle(&committed, &key, &query, decay, input, beta)?;
            let previous = std::array::from_fn(|row| &committed[row * width..(row + 1) * width]);
            let actual_output = step_eight_rows_provisional(
                previous,
                rows(&mut destination, width)?,
                &key,
                &query,
                decay,
                input,
                beta,
            )?;
            assert_eq!(bits(&committed), before);
            assert_eq!(bits(&destination), bits(&expected));
            assert_eq!(
                actual_output.map(f32::to_bits),
                expected_output.map(f32::to_bits)
            );
            std::mem::swap(&mut committed, &mut destination);
        }
    }
    Ok(())
}

#[test]
fn x1_provisional_gdn_late_failure_keeps_source_and_retries_dirty_delta() -> Result<()> {
    for width in [15, 128, 256, 257] {
        let committed = values(8 * width);
        let before = bits(&committed);
        let key = values(width);
        let query = values(width);
        let previous = std::array::from_fn(|row| &committed[row * width..(row + 1) * width]);
        let mut actual = vec![-5.0; committed.len()];
        let mut control = actual.clone();
        let mut bad_values = [0.125; 8];
        bad_values[7] = f32::NAN;
        let error = step_eight_rows_provisional(
            previous,
            rows(&mut actual, width)?,
            &key,
            &query,
            0.875,
            bad_values,
            0.625,
        )
        .err()
        .ok_or_else(|| invalid("late provisional failure was accepted"))?;
        let expected_error = step_eight_rows_from(
            previous,
            rows(&mut control, width)?,
            &key,
            &query,
            0.875,
            bad_values,
            0.625,
        )
        .err()
        .ok_or_else(|| invalid("late defensive failure was accepted"))?;
        assert_eq!(error.to_string(), expected_error.to_string());
        assert_eq!(bits(&committed), before);
        let (expected, expected_output) =
            oracle(&committed, &key, &query, 0.875, [0.125; 8], 0.625)?;
        let output = step_eight_rows_provisional(
            previous,
            rows(&mut actual, width)?,
            &key,
            &query,
            0.875,
            [0.125; 8],
            0.625,
        )?;
        assert_eq!(bits(&actual), bits(&expected));
        assert_eq!(output.map(f32::to_bits), expected_output.map(f32::to_bits));
        assert_eq!(bits(&committed), before);
    }
    Ok(())
}

#[test]
fn x1_provisional_gdn_geometry_rejects_before_destination_changes() -> Result<()> {
    let width = 128;
    let committed = values(8 * width);
    let key = values(width);
    let query = values(width - 1);
    let mut actual = vec![-5.0; committed.len()];
    let before = bits(&actual);
    assert!(
        step_eight_rows_provisional(
            std::array::from_fn(|row| &committed[row * width..(row + 1) * width]),
            rows(&mut actual, width)?,
            &key,
            &query,
            0.875,
            [0.125; 8],
            0.625,
        )
        .is_err()
    );
    assert_eq!(bits(&actual), before);
    Ok(())
}

#[test]
fn x2_provisional_head_matches_scalar_bits_and_preserves_failure_boundaries() -> Result<()> {
    for width in [128, 256] {
        let row_count = 24;
        let committed = values(row_count * width);
        let committed_bits = bits(&committed);
        let key = values(width);
        let mut query = values(width);
        query.reverse();
        let value = values(row_count);
        let scale = 0.088_388_346;
        let input = ChunkInput {
            query: &query,
            key: &key,
            value: &value,
            decay: 0.875,
            beta: 0.625,
        };
        let mut expected = committed.clone();
        let mut expected_output = vec![0.0; row_count];
        for ((state, &value), output) in expected
            .chunks_exact_mut(width)
            .zip(&value)
            .zip(&mut expected_output)
        {
            for element in &mut *state {
                *element *= input.decay;
            }
            let delta = (value - dot_scalar(state, &key)?) * input.beta;
            scalar_add_scaled(state, &key, delta)?;
            *output = dot_scalar(state, &query)? * scale;
        }
        let mut state = vec![-5.0; committed.len()];
        let mut output = vec![-7.0; row_count];
        let Some(result) =
            try_step_head_provisional(&committed, &mut state, input, scale, &mut output)
        else {
            // Feature-disabled and non-ARM targets retain the graph's old path.
            assert_eq!(bits(&state), bits(&vec![-5.0; committed.len()]));
            assert_eq!(output, vec![-7.0; row_count]);
            continue;
        };
        result?;
        assert_eq!(bits(&state), bits(&expected));
        assert_eq!(bits(&output), bits(&expected_output));
        assert_eq!(bits(&committed), committed_bits);

        let mut bad_values = value.clone();
        bad_values[15] = f32::NAN;
        state.fill(-5.0);
        output.fill(-7.0);
        let error = try_step_head_provisional(
            &committed,
            &mut state,
            ChunkInput {
                value: &bad_values,
                ..input
            },
            scale,
            &mut output,
        )
        .ok_or_else(|| invalid("eligible native head unexpectedly unavailable"))?
        .err()
        .ok_or_else(|| invalid("late head failure was accepted"))?;
        let mut control = vec![-5.0; width * 8];
        let mut bad_group = [0.0; 8];
        bad_group.copy_from_slice(&bad_values[8..16]);
        let expected_error = step_eight_rows_from(
            std::array::from_fn(|row| &committed[(8 + row) * width..(9 + row) * width]),
            rows(&mut control, width)?,
            &key,
            &query,
            input.decay,
            bad_group,
            input.beta,
        )
        .err()
        .ok_or_else(|| invalid("defensive late head failure was accepted"))?;
        assert_eq!(error.to_string(), expected_error.to_string());
        assert_eq!(bits(&output[..8]), bits(&expected_output[..8]));
        assert_eq!(output[8..], [-7.0; 16]);
        assert!(state[16 * width..].iter().all(|&value| value == -5.0));
        assert_eq!(bits(&committed), committed_bits);

        try_step_head_provisional(&committed, &mut state, input, scale, &mut output)
            .ok_or_else(|| invalid("eligible native head unexpectedly unavailable"))??;
        assert_eq!(bits(&state), bits(&expected));
        assert_eq!(bits(&output), bits(&expected_output));
    }
    Ok(())
}

#[test]
fn x2_provisional_head_geometry_rejects_before_mutation_and_keeps_fallback() -> Result<()> {
    let width = 128;
    let committed = values(8 * width);
    let key = values(width);
    let query = values(width - 1);
    let value = [0.125; 8];
    let input = ChunkInput {
        query: &query,
        key: &key,
        value: &value,
        decay: 0.875,
        beta: 0.625,
    };
    let mut state = vec![-5.0; committed.len()];
    let before = bits(&state);
    let mut output = [-7.0; 8];
    assert!(
        try_step_head_provisional(&committed, &mut state, input, 0.125, &mut output)
            .ok_or_else(|| invalid("malformed admitted head did not reject"))?
            .is_err()
    );
    assert_eq!(bits(&state), before);
    assert_eq!(output, [-7.0; 8]);
    for input in [
        ChunkInput {
            key: &key[..127],
            ..input
        },
        ChunkInput {
            value: &value[..7],
            ..input
        },
    ] {
        assert!(
            try_step_head_provisional(&committed, &mut state, input, 0.125, &mut output).is_none()
        );
        assert_eq!(bits(&state), before);
        assert_eq!(output, [-7.0; 8]);
    }
    Ok(())
}
