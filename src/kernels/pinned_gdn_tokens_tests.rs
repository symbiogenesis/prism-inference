//! Complete-head temporal admission, arithmetic, defensive replay and retry.

use super::{
    ChunkInput, dot_scalar, scalar_add_scaled, step_eight_rows_tokens_from,
    try_step_head_tokens_from,
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

fn inputs<'a>(query: &'a [f32], key: &'a [f32], value: &'a [f32]) -> Vec<ChunkInput<'a>> {
    value
        .chunks_exact(key.len())
        .enumerate()
        .map(|(token, value)| ChunkInput {
            query,
            key,
            value,
            decay: [0.875, 0.0, -0.0, -0.25][token % 4],
            beta: [0.625, 0.0, -0.0, 1.0][token % 4],
        })
        .collect()
}

fn oracle(previous: &[f32], inputs: &[ChunkInput<'_>], scale: f32) -> Result<(Vec<f32>, Vec<f32>)> {
    let width = inputs[0].key.len();
    let mut state = previous.to_vec();
    let mut output = vec![0.0; inputs.len() * width];
    for (token, input) in inputs.iter().enumerate() {
        for (row, state) in state.chunks_exact_mut(width).enumerate() {
            for element in &mut *state {
                *element *= input.decay;
            }
            let delta = (input.value[row] - dot_scalar(state, input.key)?) * input.beta;
            scalar_add_scaled(state, input.key, delta)?;
            output[token * width + row] = dot_scalar(state, input.query)? * scale;
        }
    }
    Ok((state, output))
}

fn legacy_head(
    previous: &[f32],
    state: &mut [f32],
    inputs: &[ChunkInput<'_>],
    scale: f32,
    output: &mut [f32],
) -> Result<()> {
    let width = inputs[0].key.len();
    for (group, state) in state.chunks_exact_mut(width * 8).enumerate() {
        let row = group * 8;
        let previous = std::array::from_fn(|r| &previous[(row + r) * width..(row + r + 1) * width]);
        let states = state
            .chunks_exact_mut(width)
            .collect::<Vec<_>>()
            .try_into()
            .map_err(|_| invalid("invalid legacy test geometry"))?;
        step_eight_rows_tokens_from(previous, states, inputs, row, width, scale, output)?;
    }
    Ok(())
}

#[test]
fn x1_temporal_gdn_head_matches_scalar_bits_for_all_prompt_tiles() -> Result<()> {
    for width in [128, 256] {
        let mut previous = values(width * width);
        previous[0] = -0.0;
        previous[1] = f32::from_bits(1);
        previous[2] = -f32::MIN_POSITIVE;
        let before = bits(&previous);
        let key = values(width);
        let mut query = key.clone();
        query.reverse();
        for count in [1, 3, 8, 16, 32] {
            let value = values(count * width);
            let inputs = inputs(&query, &key, &value);
            let scale = 0.088_388_346;
            let (expected, expected_output) = oracle(&previous, &inputs, scale)?;
            let mut state = vec![f32::NAN; previous.len()];
            let mut output = vec![-7.0; count * width];
            let result =
                try_step_head_tokens_from(&previous, &mut state, &inputs, scale, &mut output);
            if let Some(result) = result {
                result?;
                assert_eq!(bits(&state), bits(&expected));
                assert_eq!(bits(&output), bits(&expected_output));
            } else {
                assert!(state.iter().all(|value| value.is_nan()));
                assert_eq!(bits(&output), bits(&vec![-7.0; count * width]));
            }
            assert_eq!(bits(&previous), before);
        }
    }
    Ok(())
}

#[test]
fn x1_temporal_gdn_head_failure_replays_legacy_publication_and_retries() -> Result<()> {
    for width in [128, 256] {
        let previous = values(width * width);
        let before = bits(&previous);
        let key = values(width);
        let query = values(width);
        for (bad_row, earlier_token_row) in [(15, 16), (8, 15)] {
            let mut value = values(3 * width);
            value[2 * width + bad_row] = f32::NAN;
            value[earlier_token_row] = f32::NAN;
            let bad_inputs = inputs(&query, &key, &value);
            let mut state = vec![-5.0; previous.len()];
            let mut output = vec![-7.0; 3 * width];
            let Some(result) =
                try_step_head_tokens_from(&previous, &mut state, &bad_inputs, 0.125, &mut output)
            else {
                assert_eq!(bits(&state), bits(&vec![-5.0; previous.len()]));
                assert_eq!(bits(&output), bits(&vec![-7.0; 3 * width]));
                continue;
            };
            let error = result
                .err()
                .ok_or_else(|| invalid("late temporal failure was accepted"))?;
            let mut control = vec![-5.0; previous.len()];
            let mut control_output = vec![-7.0; 3 * width];
            let expected_error = legacy_head(
                &previous,
                &mut control,
                &bad_inputs,
                0.125,
                &mut control_output,
            )
            .err()
            .ok_or_else(|| invalid("legacy temporal failure was accepted"))?;
            assert_eq!(error.to_string(), expected_error.to_string());
            assert_eq!(bits(&state), bits(&control));
            assert_eq!(bits(&output), bits(&control_output));
            assert_eq!(bits(&previous), before);

            value[2 * width + bad_row] = 0.125;
            value[earlier_token_row] = 0.125;
            let good_inputs = inputs(&query, &key, &value);
            let (expected, expected_output) = oracle(&previous, &good_inputs, 0.125)?;
            try_step_head_tokens_from(&previous, &mut state, &good_inputs, 0.125, &mut output)
                .ok_or_else(|| invalid("admitted temporal head unexpectedly unavailable"))??;
            assert_eq!(bits(&state), bits(&expected));
            assert_eq!(bits(&output), bits(&expected_output));
            assert_eq!(bits(&previous), before);
        }
    }
    Ok(())
}

#[test]
fn x1_temporal_gdn_head_rejects_geometry_before_mutation_and_keeps_fallback() -> Result<()> {
    let width = 128;
    let previous = values(width * width);
    let key = values(width);
    let query = values(width);
    let value = values(3 * width);
    let mut inputs = inputs(&query, &key, &value);
    inputs[2].value = &value[..width - 1];
    let mut state = vec![-5.0; previous.len()];
    let before = bits(&state);
    let mut output = vec![-7.0; 3 * width];
    assert!(
        try_step_head_tokens_from(&previous, &mut state, &inputs, 0.125, &mut output)
            .ok_or_else(|| invalid("malformed eligible head did not reject"))?
            .is_err()
    );
    assert_eq!(bits(&state), before);
    assert_eq!(bits(&output), bits(&vec![-7.0; 3 * width]));
    for unsupported in [
        Vec::new(),
        vec![ChunkInput {
            key: &key[..127],
            ..inputs[0]
        }],
        vec![inputs[0]; 65],
    ] {
        assert!(
            try_step_head_tokens_from(&previous, &mut state, &unsupported, 0.125, &mut output)
                .is_none()
        );
        assert_eq!(bits(&state), before);
        assert_eq!(bits(&output), bits(&vec![-7.0; 3 * width]));
    }
    Ok(())
}
