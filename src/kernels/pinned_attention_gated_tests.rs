//! Independent channel-wise reduction and exceptional-publication controls.

use bonsai_ternary::half_to_float;

use super::{
    dot, value_tile_four, value_tile_four_gated, value_tile_four_two_tokens,
    value_tile_four_two_tokens_gated,
};

fn output_refs(outputs: &mut [Vec<f32>; 4]) -> [&mut [f32]; 4] {
    let [a, b, c, d] = outputs;
    [a, b, c, d]
}

fn weights(length: usize) -> [Vec<f32>; 4] {
    std::array::from_fn(|head| {
        (0..length)
            .map(|token| half_to_float([0, 0x8000, 0x0001, 0x3400, 0xb800][(token + head) % 5]))
            .collect()
    })
}

fn gates(width: usize) -> [Vec<f32>; 4] {
    std::array::from_fn(|head| {
        (0..width)
            .map(|channel| [-100.0, -4.0, -0.0, 0.0, 1.0, 100.0][(channel + head) % 6])
            .collect()
    })
}

fn assert_bits(actual: &[Vec<f32>; 4], expected: &[Vec<f32>; 4]) {
    for (actual, expected) in actual.iter().zip(expected) {
        assert!(actual.iter().map(|v| v.to_bits()).eq(expected.iter().map(|v| v.to_bits())));
    }
}

fn reference(
    rows: &[Vec<u16>],
    current: &[f32],
    weights: &[Vec<f32>; 4],
    gates: &[Vec<f32>; 4],
    offset: usize,
) -> crate::Result<[Vec<f32>; 4]> {
    let mut output = std::array::from_fn(|_| vec![0.0; current.len()]);
    for (head, out) in output.iter_mut().enumerate() {
        for (channel, &value) in current.iter().enumerate() {
            let reduction = dot(
                rows.iter()
                    .map(|row| half_to_float(row[offset + channel]))
                    .chain(std::iter::once(value)),
                &weights[head],
            )?;
            out[channel] = reduction * (1.0 / (1.0 + (-gates[head][channel]).exp()));
        }
    }
    Ok(output)
}

#[test]
fn gated_four_head_tiles_match_independent_channel_oracle() -> crate::Result<()> {
    let pattern = [0, 0x8000, 0x0001, 0x8001, 0x3c00, 0xbc00, 0x3555, 0xb555];
    for length in [1, 2, 15, 16, 17, 63, 64, 65, 255, 256, 257] {
        for width in [1, 3, 4, 5, 7, 8, 31, 32, 63, 64, 127, 128] {
            let offset = 3;
            let rows: Vec<Vec<u16>> = (0..length - 1)
                .map(|token| (0..offset + width).map(|channel| pattern[(token + channel) % 8]).collect())
                .collect();
            let current: Vec<f32> = (0..width).map(|channel| half_to_float(pattern[channel % 8])).collect();
            let weights = weights(length);
            let gates = gates(width);
            let expected = reference(&rows, &current, &weights, &gates, offset)?;
            let mut actual = std::array::from_fn(|_| vec![19.0; width]);
            value_tile_four_gated(
                rows.iter().map(Vec::as_slice),
                &current,
                weights.each_ref().map(Vec::as_slice),
                offset,
                output_refs(&mut actual),
                gates.each_ref().map(Vec::as_slice),
            )?;
            assert_bits(&actual, &expected);
        }
    }
    Ok(())
}

#[test]
fn gated_two_token_tiles_match_independent_causal_oracle() -> crate::Result<()> {
    for length in [1, 2, 15, 16, 63, 64, 255, 256] {
        for width in [1, 3, 4, 5, 8, 31, 32, 63, 64] {
            let rows: Vec<Vec<u16>> = (0..length)
                .map(|token| (0..width).map(|channel| [0x3555, 0xb800, 0, 0x8000][(token + channel) % 4]).collect())
                .collect();
            let current0: Vec<f32> = rows[length - 1].iter().copied().map(half_to_float).collect();
            let current1: Vec<f32> = (0..width).map(|channel| [0.0, -0.0, 0.5, -0.25][channel % 4]).collect();
            let w0 = weights(length);
            let w1 = weights(length + 1);
            let g0 = gates(width);
            let mut g1 = gates(width);
            g1.rotate_left(1);
            let expected0 = reference(&rows[..length - 1], &current0, &w0, &g0, 0)?;
            let expected1 = reference(&rows, &current1, &w1, &g1, 0)?;
            let mut actual0 = std::array::from_fn(|_| vec![19.0; width]);
            let mut actual1 = std::array::from_fn(|_| vec![19.0; width]);
            value_tile_four_two_tokens_gated(
                rows.iter().map(Vec::as_slice),
                [&current0, &current1],
                w0.each_ref().map(Vec::as_slice),
                w1.each_ref().map(Vec::as_slice),
                0,
                output_refs(&mut actual0),
                output_refs(&mut actual1),
                [g0.each_ref().map(Vec::as_slice), g1.each_ref().map(Vec::as_slice)],
            )?;
            assert_bits(&actual0, &expected0);
            assert_bits(&actual1, &expected1);
        }
    }
    Ok(())
}

#[test]
fn gated_tiles_preserve_ungated_failure_publication_and_reject_bad_gates() {
    for width in [1, 3, 4, 8, 128] {
        for bad_channel in [0, width - 1] {
            for bad in [f32::MAX, f32::NAN, f32::INFINITY] {
                let mut current = vec![1.0; width];
                current[bad_channel] = bad;
                let weights = [vec![2.0], vec![0.25], vec![1.0], vec![0.5]];
                let gates = gates(width);
                let mut expected = std::array::from_fn(|_| vec![19.0; width]);
                let mut actual = expected.clone();
                assert!(value_tile_four(
                    std::iter::empty(), &current, weights.each_ref().map(Vec::as_slice),
                    0, output_refs(&mut expected),
                ).is_err());
                assert!(value_tile_four_gated(
                    std::iter::empty(), &current, weights.each_ref().map(Vec::as_slice),
                    0, output_refs(&mut actual), gates.each_ref().map(Vec::as_slice),
                ).is_err());
                assert_bits(&actual, &expected);
                // Independent legacy contract: the first head publishes its
                // finite prefix, then stops before the failing channel.
                assert!(actual[0][..bad_channel].iter().all(|value| value.to_bits() == 2.0_f32.to_bits()));
                assert!(actual[0][bad_channel..].iter().all(|value| value.to_bits() == 19.0_f32.to_bits()));
                assert!(actual[1..].iter().flatten().all(|value| value.to_bits() == 19.0_f32.to_bits()));
            }
        }
    }
    let mut output = std::array::from_fn(|_| vec![19.0; 4]);
    assert!(value_tile_four_gated(
        std::iter::empty(), &[1.0; 4], [&[1.0]; 4], 0,
        output_refs(&mut output), [&[0.0; 3]; 4],
    ).is_err());
    assert_eq!(output, [vec![19.0; 4], vec![19.0; 4], vec![19.0; 4], vec![19.0; 4]]);
}

#[test]
fn gated_two_token_tiles_never_gate_a_failed_partial_publication() {
    for width in [3, 4, 8, 64] {
        for failing_token in 0..2 {
            let mut currents = [vec![1.0; width], vec![1.0; width]];
            currents[failing_token][width - 1] = f32::MAX;
            let weights0 = [vec![0.0], vec![0.25], vec![0.5], vec![2.0]];
            let weights1 = [vec![0.0, 0.0], vec![0.25, 0.25], vec![0.5, 0.5], vec![2.0, 2.0]];
            let gate = gates(width);
            let rows = [vec![0x3c00_u16; width]];
            let mut expected0 = std::array::from_fn(|_| vec![19.0; width]);
            let mut expected1 = expected0.clone();
            let mut actual0 = expected0.clone();
            let mut actual1 = expected0.clone();
            assert!(value_tile_four_two_tokens(
                rows.iter().map(Vec::as_slice),
                [&currents[0], &currents[1]],
                weights0.each_ref().map(Vec::as_slice), weights1.each_ref().map(Vec::as_slice), 0,
                output_refs(&mut expected0), output_refs(&mut expected1),
            ).is_err());
            assert!(value_tile_four_two_tokens_gated(
                rows.iter().map(Vec::as_slice),
                [&currents[0], &currents[1]],
                weights0.each_ref().map(Vec::as_slice), weights1.each_ref().map(Vec::as_slice), 0,
                output_refs(&mut actual0), output_refs(&mut actual1),
                [gate.each_ref().map(Vec::as_slice); 2],
            ).is_err());
            assert_bits(&actual0, &expected0);
            assert_bits(&actual1, &expected1);
        }
    }
}
