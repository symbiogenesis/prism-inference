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
    struct WeightedValue {
        value_half_bits: Vec<u16>,
        probability_half_bits: Vec<u16>,
        native_bits: u32,
    }

    #[test]
    fn neon_value_tile_baseline_is_available_and_accumulator_remains_two_kib() -> crate::Result<()> {
        let neon = fearless_simd::Level::baseline()
            .as_neon()
            .ok_or_else(|| crate::invalid("required attention baseline NEON token is absent"))?;
        assert_eq!(size_of::<[[f32x4<Neon>; VALUE_TILE / 4]; 16]>(), 2048);
        let capture: Capture = serde_json::from_str(include_str!(
            "../../tests/fixtures/swift-attention-dot.json"
        ))?;
        let captured = capture.weighted_value;
        assert!(!captured.value_half_bits.is_empty());
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
        assert!(output.iter().all(|value| value.to_bits() == captured.native_bits));
        Ok(())
    }
}
