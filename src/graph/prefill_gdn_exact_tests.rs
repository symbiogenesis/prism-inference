//! Independent pinned arithmetic at the integrated temporal consumer boundary.

use super::{Arithmetic, Input, exact_head_into, small_float};
use crate::{Result, invalid};

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

fn values(count: usize, mut seed: u32) -> Vec<f32> {
    (0..count)
        .map(|_| {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            f32::from_bits((seed & 0x807f_ffff) | 0x3c00_0000)
        })
        .collect()
}

fn finite(value: f32) -> Result<f32> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(invalid("nonfinite independent GDN result"))
    }
}

#[expect(
    clippy::suboptimal_flops,
    reason = "The independent pinned oracle preserves separate scalar-tail rounding."
)]
fn dot(first: &[f32], second: &[f32]) -> Result<f32> {
    // The pinned contract has four FMA registers, each with four temporal
    // lanes. Preserve its register tree and adjacent-pair horizontal sum.
    let aligned = first.len() / 16 * 16;
    let mut sums = [[0.0_f32; 4]; 4];
    for offset in (0..aligned).step_by(16) {
        for (register, lanes) in sums.iter_mut().enumerate() {
            for (lane, sum) in lanes.iter_mut().enumerate() {
                let coordinate = offset + register * 4 + lane;
                *sum = first[coordinate].mul_add(second[coordinate], *sum);
            }
        }
    }
    let lanes: [f32; 4] = std::array::from_fn(|lane| {
        (sums[0][lane] + sums[2][lane]) + (sums[1][lane] + sums[3][lane])
    });
    let mut sum = (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);
    for (&first, &second) in first[aligned..].iter().zip(&second[aligned..]) {
        sum += first * second;
    }
    finite(sum)
}

#[expect(
    clippy::suboptimal_flops,
    reason = "The independent recurrence keeps separate scalar-tail update rounding."
)]
fn oracle(previous: &[f32], inputs: &[Input<'_>]) -> Result<(Vec<f32>, Vec<f32>)> {
    let width = inputs[0].key.len();
    let scale = 1.0 / small_float(width)?.sqrt();
    let mut state = previous.to_vec();
    let mut output = vec![0.0; inputs.len() * width];
    // Token-major scalar recurrence is deliberately independent of native
    // eight-row staging, head admission and consumer dispatch.
    for (token, input) in inputs.iter().enumerate() {
        for (row, state) in state.chunks_exact_mut(width).enumerate() {
            for value in &mut *state {
                *value *= input.decay;
            }
            let delta = (input.value[row] - dot(state, input.key)?) * input.beta;
            let aligned = width / 16 * 16;
            for (column, (value, &key)) in state.iter_mut().zip(input.key).enumerate() {
                *value = finite(if column < aligned {
                    delta.mul_add(key, *value)
                } else {
                    *value + delta * key
                })?;
            }
            output[token * width + row] = dot(state, input.query)? * scale;
        }
    }
    Ok((state, output))
}

fn inputs<'a>(width: usize, query: &'a [f32], key: &'a [f32], value: &'a [f32]) -> Vec<Input<'a>> {
    key.chunks_exact(width)
        .zip(query.chunks_exact(width))
        .zip(value.chunks_exact(width))
        .enumerate()
        .map(|(token, ((key, query), value))| Input {
            key,
            query,
            value,
            decay: [0.875, 0.75, -0.25, 0.0, -0.0][token % 5],
            beta: [0.625, 0.375, 1.0, -0.0, 0.0][token % 5],
        })
        .collect()
}

#[test]
fn x1_exact_gdn_consumer_matches_independent_scalar_all_tiles_and_fallback() -> Result<()> {
    for width in [15, 128, 256] {
        let mut previous = values(width * width, 19);
        previous[0] = -0.0;
        previous[1] = f32::from_bits(1);
        previous[2] = -f32::MIN_POSITIVE;
        for count in [1, 3, 8, 16, 32] {
            let query = values(count * width, 23);
            let key = values(count * width, 29);
            let value = values(count * width, 31);
            let inputs = inputs(width, &query, &key, &value);
            let before = bits(&previous);
            let (expected, expected_output) = oracle(&previous, &inputs)?;
            let mut state = vec![f32::NAN; previous.len()];
            let mut output = vec![f32::NAN; count * width];
            exact_head_into(
                Arithmetic::PinnedSwift,
                &previous,
                &inputs,
                &mut state,
                &mut output,
            )?;
            assert_eq!(
                bits(&state),
                bits(&expected),
                "width={width}, count={count}"
            );
            assert_eq!(bits(&output), bits(&expected_output));
            assert_eq!(bits(&previous), before);
        }
    }
    Ok(())
}

#[test]
fn x1_exact_gdn_consumer_late_failure_preserves_source_and_recovers_dirty_delta() -> Result<()> {
    for width in [15, 128, 256] {
        let count = 8;
        let previous = values(width * width, 37);
        let before = bits(&previous);
        let query = values(count * width, 41);
        let key = values(count * width, 43);
        let mut value = values(count * width, 47);
        let good_inputs = inputs(width, &query, &key, &value);
        let (expected, expected_output) = oracle(&previous, &good_inputs)?;
        let bad_index = (count - 1) * width + width - 1;
        let good_value = value[bad_index];
        value[bad_index] = f32::NAN;
        let bad_inputs = inputs(width, &query, &key, &value);
        let mut state = vec![-5.0; previous.len()];
        let mut output = vec![-7.0; count * width];
        assert!(
            exact_head_into(
                Arithmetic::PinnedSwift,
                &previous,
                &bad_inputs,
                &mut state,
                &mut output,
            )
            .is_err()
        );
        assert_eq!(bits(&previous), before);
        // At least one preceding group completed before the final-row error.
        assert_eq!(bits(&state[..8 * width]), bits(&expected[..8 * width]));
        value[bad_index] = good_value;
        let good_inputs = inputs(width, &query, &key, &value);
        exact_head_into(
            Arithmetic::PinnedSwift,
            &previous,
            &good_inputs,
            &mut state,
            &mut output,
        )?;
        assert_eq!(bits(&state), bits(&expected));
        assert_eq!(bits(&output), bits(&expected_output));
        assert_eq!(bits(&previous), before);
    }
    Ok(())
}
