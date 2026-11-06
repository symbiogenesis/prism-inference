//! Exact baseline NEON value tiles with adjacent channels in vector lanes.

use fearless_simd::{Neon, f32x4, prelude::*};

use crate::{Result, invalid};

use super::{VALUE_TILE, padded_dot_steps};

/// Consumes a validated tile whose width is a positive multiple of four.
///
/// The sixteen native temporal FMA lanes stay independent. Each vector groups
/// four adjacent value channels, making an update contiguous without changing
/// any channel's token order. The accumulator remains exactly 2 KiB; decoded
/// FP16 values exist only for one four-channel load. No FP16 arithmetic or
/// optional instruction set is required.
pub(super) fn value_tile<'a>(
    neon: Neon,
    mut past: impl Iterator<Item = &'a [u16]>,
    current: &[f32],
    weights: &[f32],
    range: &std::ops::Range<usize>,
    output: &mut [f32],
) -> Result<()> {
    let zero = f32x4::splat(neon, 0.0);
    let mut sums = [[zero; VALUE_TILE / 4]; 16];
    let groups = output.len() / 4;
    let last = weights.len() - 1;
    for (index, &weight) in weights[..last].iter().enumerate() {
        let values = past
            .next()
            .and_then(|row| row.get(range.start..range.end))
            .ok_or_else(|| invalid("pinned attention value row is short"))?;
        let weight = f32x4::splat(neon, weight);
        for (sum, values) in sums[index % 16][..groups]
            .iter_mut()
            .zip(values.as_chunks::<4>().0)
        {
            let values = f32x4::load_array(neon, (*values).map(bonsai_ternary::half_to_float));
            *sum = values.mul_add_precise(weight, *sum);
        }
    }
    if past.next().is_some() {
        return Err(invalid("pinned attention value row is long"));
    }
    let weight = f32x4::splat(neon, weights[last]);
    for (sum, values) in sums[last % 16][..groups]
        .iter_mut()
        .zip(current.as_chunks::<4>().0)
    {
        *sum = f32x4::load_array(neon, *values).mul_add_precise(weight, *sum);
    }
    // Preserve the first masked FMA in every temporal lane, including -0.
    for index in weights.len()..padded_dot_steps(weights.len()) {
        for sum in &mut sums[index % 16][..groups] {
            *sum = zero.mul_add_precise(zero, *sum);
        }
    }
    for (group, out) in output.as_chunks_mut::<4>().0.iter_mut().enumerate() {
        let lanes: [f32x4<Neon>; 4] = std::array::from_fn(|lane| {
            (sums[lane][group] + sums[8 + lane][group])
                + (sums[4 + lane][group] + sums[12 + lane][group])
        });
        let result = ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3])).to_array();
        // Publication and finite rejection retain increasing channel order.
        for (out, value) in out.iter_mut().zip(result) {
            *out = super::super::finite(value)?;
        }
    }
    Ok(())
}

/// Reduces a validated F16 key slice and F32 query slice with four NEON FMA registers.
pub(super) fn dot_f16(neon: Neon, key: &[u16], query: &[f32]) -> Result<f32> {
    let mut sums = [f32x4::splat(neon, 0.0); 4];
    let key_chunks = key.as_chunks::<4>().0;
    let query_chunks = query.as_chunks::<4>().0;
    for (chunk, (k, q)) in key_chunks.iter().zip(query_chunks).enumerate() {
        let acc = &mut sums[chunk % 4];
        let kv = f32x4::load_array(neon, (*k).map(bonsai_ternary::half_to_float));
        let qv = f32x4::load_array(neon, *q);
        *acc = kv.mul_add_precise(qv, *acc);
    }
    let vector_steps = key_chunks.len() * 4;
    let total_steps = padded_dot_steps(query.len());
    if vector_steps == total_steps {
        let lanes = ((sums[0] + sums[2]) + (sums[1] + sums[3])).to_array();
        return super::super::finite((lanes[0] + lanes[1]) + (lanes[2] + lanes[3]));
    }
    let mut scalar_sums = sums.map(f32x4::to_array);
    for index in vector_steps..total_steps {
        let (value, weight) = if index < query.len() {
            (bonsai_ternary::half_to_float(key[index]), query[index])
        } else {
            (0.0, 0.0)
        };
        let sum = &mut scalar_sums[index % 16 / 4][index % 4];
        *sum = value.mul_add(weight, *sum);
    }
    let mut lanes = [0.0; 4];
    for (index, lane) in lanes.iter_mut().enumerate() {
        *lane = (scalar_sums[0][index] + scalar_sums[2][index])
            + (scalar_sums[1][index] + scalar_sums[3][index]);
    }
    super::super::finite((lanes[0] + lanes[1]) + (lanes[2] + lanes[3]))
}

/// Reduces a validated F32 key slice and F32 query slice with four NEON FMA registers.
pub(super) fn dot_f32(neon: Neon, key: &[f32], query: &[f32]) -> Result<f32> {
    let mut sums = [f32x4::splat(neon, 0.0); 4];
    let key_chunks = key.as_chunks::<4>().0;
    let query_chunks = query.as_chunks::<4>().0;
    for (chunk, (k, q)) in key_chunks.iter().zip(query_chunks).enumerate() {
        let acc = &mut sums[chunk % 4];
        let kv = f32x4::load_array(neon, *k);
        let qv = f32x4::load_array(neon, *q);
        *acc = kv.mul_add_precise(qv, *acc);
    }
    let vector_steps = key_chunks.len() * 4;
    let total_steps = padded_dot_steps(query.len());
    if vector_steps == total_steps {
        let lanes = ((sums[0] + sums[2]) + (sums[1] + sums[3])).to_array();
        return super::super::finite((lanes[0] + lanes[1]) + (lanes[2] + lanes[3]));
    }
    let mut scalar_sums = sums.map(f32x4::to_array);
    for index in vector_steps..total_steps {
        let (value, weight) = if index < query.len() {
            (key[index], query[index])
        } else {
            (0.0, 0.0)
        };
        let sum = &mut scalar_sums[index % 16 / 4][index % 4];
        *sum = value.mul_add(weight, *sum);
    }
    let mut lanes = [0.0; 4];
    for (index, lane) in lanes.iter_mut().enumerate() {
        *lane = (scalar_sums[0][index] + scalar_sums[2][index])
            + (scalar_sums[1][index] + scalar_sums[3][index]);
    }
    super::super::finite((lanes[0] + lanes[1]) + (lanes[2] + lanes[3]))
}

/// Rotates validated 256-wide Swift heads (with 64-element rotary prefix) in NEON `f32x4`.
pub(super) fn rope_with_aliasing(
    neon: Neon,
    input: &mut [f32],
    frequencies: &[(f32, f32); 32],
    in_place: bool,
) {
    let cos_vecs: [f32x4<Neon>; 8] = std::array::from_fn(|chunk| {
        let base = chunk * 4;
        f32x4::load_array(
            neon,
            [
                frequencies[base].0,
                frequencies[base + 1].0,
                frequencies[base + 2].0,
                frequencies[base + 3].0,
            ],
        )
    });
    let sin_vecs: [f32x4<Neon>; 8] = std::array::from_fn(|chunk| {
        let base = chunk * 4;
        f32x4::load_array(
            neon,
            [
                frequencies[base].1,
                frequencies[base + 1].1,
                frequencies[base + 2].1,
                frequencies[base + 3].1,
            ],
        )
    });
    for head in input.as_chunks_mut::<256>().0 {
        let (first_half, second_half) = head[..64].split_at_mut(32);
        for (((first_chunk, second_chunk), &cos), &sin) in first_half
            .as_chunks_mut::<4>()
            .0
            .iter_mut()
            .zip(second_half.as_chunks_mut::<4>().0)
            .zip(&cos_vecs)
            .zip(&sin_vecs)
        {
            let first = f32x4::from_slice(neon, first_chunk);
            let second = f32x4::from_slice(neon, second_chunk);
            let rotated_first = if in_place {
                first.mul_add_precise(cos, -(second * sin))
            } else {
                (-second).mul_add_precise(sin, first * cos)
            };
            let rotated_second = second.mul_add_precise(cos, first * sin);
            rotated_first.store_slice(first_chunk);
            rotated_second.store_slice(second_chunk);
        }
    }
}

#[cfg(test)]
mod tests {
    use bonsai_ternary::half_to_float;
    use fearless_simd::{Neon, f32x4};

    use super::{VALUE_TILE, value_tile};

    #[derive(serde::Deserialize)]
    struct Capture {
        weighted_value: WeightedValue,
    }

    #[derive(serde::Deserialize)]
    #[allow(
        clippy::struct_field_names,
        reason = "Field names match the immutable JSON fixture schema."
    )]
    struct WeightedValue {
        value_half_bits: Vec<u16>,
        probability_half_bits: Vec<u16>,
        native_bits: u32,
    }

    #[test]
    fn neon_value_tile_baseline_is_available_and_accumulator_remains_two_kib() -> crate::Result<()>
    {
        let neon = fearless_simd::Level::baseline()
            .as_neon()
            .ok_or_else(|| crate::invalid("required attention baseline NEON token is absent"))?;
        assert_eq!(size_of::<[[f32x4<Neon>; VALUE_TILE / 4]; 16]>(), 2048);
        let capture: Capture = serde_json::from_str(include_str!(
            "../../tests/fixtures/swift-attention-dot.json"
        ))?;
        let captured = capture.weighted_value;
        assert_ne!(captured.value_half_bits, [0_u16; 0]);
        let last = captured.value_half_bits.len() - 1;
        let rows: Vec<_> = captured.value_half_bits[..last]
            .iter()
            .map(|&value| [value; VALUE_TILE])
            .collect();
        let current = [half_to_float(captured.value_half_bits[last]); VALUE_TILE];
        let weights: Vec<_> = captured
            .probability_half_bits
            .into_iter()
            .map(half_to_float)
            .collect();
        let mut output = [f32::NAN; VALUE_TILE];
        value_tile(
            neon,
            rows.iter().map(<[u16; VALUE_TILE]>::as_slice),
            &current,
            &weights,
            &(0..VALUE_TILE),
            &mut output,
        )?;
        assert!(
            output
                .iter()
                .all(|value| value.to_bits() == captured.native_bits)
        );
        Ok(())
    }
}
