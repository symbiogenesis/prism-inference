//! Independent RMS arithmetic plus defensive-failure controls for scoped weights.

use super::RmsNormWeights;
use crate::{Result, kernels};

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

#[expect(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "The independent normalization oracle explicitly rounds its binary64 mean to binary32."
)]
fn reference<const PINNED: bool>(
    input: &[f32],
    weight: &[f32],
    gate: &[f32],
    epsilon: f32,
) -> Result<Vec<f32>> {
    let mut squares = 0.0_f64;
    for &value in input {
        squares += f64::from(value * value);
    }
    let mean = (squares / f64::from(u32::try_from(input.len())?)) as f32;
    let scale = 1.0 / (mean + epsilon).sqrt();
    let mut output: Vec<_> = input
        .iter()
        .zip(weight)
        .map(|(&value, &w)| {
            if PINNED {
                (value * w) * scale
            } else {
                (value * scale) * w
            }
        })
        .collect();
    if PINNED {
        // Activation has separate native/scalar controls. This oracle shares no
        // normalization, geometry or admission code with the scoped consumer.
        kernels::pinned_activations::swiglu_in_place(gate, &mut output)?;
    } else {
        for (value, &g) in output.iter_mut().zip(gate) {
            *value *= g / (1.0 + (-g).exp());
        }
    }
    Ok(output)
}

fn repeated_heads<const PINNED: bool>() -> Result<()> {
    let samples = [
        1.5,
        -2.25,
        0.0,
        -0.0,
        f32::from_bits(1),
        -f32::from_bits(1),
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
        f32::from_bits(0x3f80_0001),
    ];
    for width in [1, 3, 4, 7, 15, 16, 17, 128, 256] {
        let weights: Vec<_> = (0..width)
            .map(|index| [1.25, -0.75, -0.0, 0.0, 0.5][index % 5])
            .collect();
        let original_weights = bits(&weights);
        let scope = RmsNormWeights::new(&weights)?;
        for head in 0..48 {
            let input: Vec<_> = (0..width)
                .map(|index| samples[(index + head) % samples.len()])
                .collect();
            let gate: Vec<_> = (0..width)
                .map(|index| [-8.0, -0.0, 0.0, 0.75, 8.0][(index + head) % 5])
                .collect();
            let expected = reference::<PINNED>(&input, &weights, &gate, 0.000_001)?;
            let mut actual = input.clone();
            scope.swiglu_in_place::<PINNED>(&mut actual, 0.000_001, &gate)?;
            assert_eq!(bits(&actual), bits(&expected), "width {width}, head {head}");
            let mut output = vec![f32::NAN; width];
            scope.swiglu_into::<PINNED>(&input, 0.000_001, &gate, &mut output)?;
            assert_eq!(bits(&output), bits(&expected), "width {width}, head {head}");
        }
        assert_eq!(bits(&weights), original_weights);
    }
    Ok(())
}

#[test]
fn recurrent_rms_weight_scope_preserves_independent_source_and_pinned_head_bits() -> Result<()> {
    repeated_heads::<false>()?;
    repeated_heads::<true>()
}

fn compare_failure<const PINNED: bool>(
    input: &[f32],
    weights: &[f32],
    epsilon: f32,
    gate: &[f32],
    output_len: usize,
) -> Result<()> {
    let scope = RmsNormWeights::new(weights)?;
    let mut expected = input.to_vec();
    let mut actual = input.to_vec();
    let expected_result =
        kernels::rms_norm_swiglu_in_place::<PINNED>(&mut expected, weights, epsilon, gate);
    let actual_result = scope.swiglu_in_place::<PINNED>(&mut actual, epsilon, gate);
    assert_eq!(
        actual_result.as_ref().err().map(ToString::to_string),
        expected_result.as_ref().err().map(ToString::to_string)
    );
    assert_eq!(bits(&actual), bits(&expected));
    let mut expected = vec![f32::from_bits(0x7fc0_0023); output_len];
    let mut actual = expected.clone();
    let expected_result =
        kernels::rms_norm_swiglu_into::<PINNED>(input, weights, epsilon, gate, &mut expected);
    let actual_result = scope.swiglu_into::<PINNED>(input, epsilon, gate, &mut actual);
    assert_eq!(
        actual_result.as_ref().err().map(ToString::to_string),
        expected_result.as_ref().err().map(ToString::to_string)
    );
    assert_eq!(bits(&actual), bits(&expected));
    Ok(())
}

fn rejected_inputs<const PINNED: bool>() -> Result<()> {
    for width in [1, 3, 4, 15, 16, 17, 128, 256] {
        let weights = vec![1.0; width];
        let gate = vec![0.75; width];
        for index in [0, width / 2, width - 1] {
            for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, f32::MAX] {
                let mut input = vec![0.5; width];
                input[index] = bad;
                compare_failure::<PINNED>(&input, &weights, 0.000_001, &gate, width)?;
            }
        }
        let input = vec![0.5; width];
        for epsilon in [f32::NAN, f32::INFINITY, -1.0] {
            compare_failure::<PINNED>(&input, &weights, epsilon, &gate, width)?;
        }
        compare_failure::<PINNED>(&input, &weights[..width - 1], 0.0, &gate, width)?;
        compare_failure::<PINNED>(&input, &weights, 0.0, &gate[..width - 1], width)?;
        compare_failure::<PINNED>(&input, &weights, 0.0, &gate, width - 1)?;
        compare_failure::<PINNED>(&vec![0.0; width], &weights, 0.0, &gate, width)?;
        let mut weights = weights;
        weights[width - 1] = f32::MAX;
        compare_failure::<PINNED>(&vec![2.0; width], &weights, 0.0, &gate, width)?;
    }
    compare_failure::<PINNED>(&[], &[], 0.0, &[], 0)
}

#[test]
fn recurrent_rms_weight_scope_retains_per_call_rejection_and_failure_mutation() -> Result<()> {
    rejected_inputs::<false>()?;
    rejected_inputs::<true>()
}

#[test]
fn recurrent_rms_weight_scope_rejects_nonfinite_weights_and_never_reuses_stale_admission()
-> Result<()> {
    for index in [0, 63, 127] {
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut weights = [1.0; 128];
            weights[index] = bad;
            assert!(RmsNormWeights::new(&weights).is_err());
            weights[index] = -0.0;
            let scope = RmsNormWeights::new(&weights)?;
            let mut output = [0.5; 128];
            scope.swiglu_in_place::<true>(&mut output, 0.000_001, &[0.75; 128])?;
            assert!(output.iter().all(|value| value.is_finite()));
            // The immutable borrow has ended; a newly mutated owner requires
            // fresh admission rather than inheriting a pointer-keyed verdict.
            weights[index] = bad;
            assert!(RmsNormWeights::new(&weights).is_err());
        }
    }
    Ok(())
}
