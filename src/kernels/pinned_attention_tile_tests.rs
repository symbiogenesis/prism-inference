//! Native-width value-tile controls against the retained scalar arithmetic.

use bonsai_ternary::half_to_float;

use super::{VALUE_TILE, value_tile, value_tile_scalar};

#[test]
fn pinned_value_vector_tiles_preserve_scalar_bits_at_channel_and_padding_boundaries()
-> crate::Result<()> {
    let pattern = [
        0_u16, 0x8000, 0x0001, 0x8001, 0x3c00, 0xbc00, 0x3555, 0xb555, 0x7bff, 0xfbff,
    ];
    for length in [1, 3, 4, 15, 16, 17, 255, 256, 257, 8191, 8192] {
        for width in [1, 3, 4, 5, 7, 8, 12, 16, 28, 31, VALUE_TILE] {
            let offset = 3;
            let rows: Vec<Vec<u16>> = (0..length)
                .map(|token| {
                    (0..offset + width)
                        .map(|channel| pattern[(token * 3 + channel) % pattern.len()])
                        .collect()
                })
                .collect();
            let current: Vec<_> = rows[length - 1][offset..]
                .iter()
                .copied()
                .map(half_to_float)
                .collect();
            let weights: Vec<_> = (0..length)
                .map(|token| {
                    half_to_float([0x0000, 0x8000, 0x0001, 0x3c00, 0xb800][token % 5])
                })
                .collect();
            let past = rows[..length - 1].iter().map(Vec::as_slice);
            let mut actual = vec![f32::NAN; width];
            let mut expected = actual.clone();
            value_tile(past.clone(), &current, &weights, offset, &mut actual)?;
            value_tile_scalar(
                past,
                &current,
                &weights,
                &(offset..offset + width),
                &mut expected,
            )?;
            assert!(
                actual
                    .iter()
                    .zip(expected)
                    .all(|(actual, expected)| actual.to_bits() == expected.to_bits()),
                "length {length}, width {width}"
            );
        }
    }
    Ok(())
}

#[test]
fn pinned_value_vector_tiles_preserve_validation_and_finite_failure_publication() {
    let rows = [[0x3c00_u16; 4]; 2];
    let mut actual = [19.0; 4];
    assert!(
        value_tile(
            rows[..1].iter().map(<[u16; 4]>::as_slice),
            &[1.0; 4],
            &[1.0; 3],
            0,
            &mut actual,
        )
        .is_err()
    );
    assert_eq!(actual, [19.0; 4]);
    assert!(
        value_tile(
            rows.iter().map(<[u16; 4]>::as_slice),
            &[1.0; 4],
            &[1.0; 2],
            0,
            &mut actual,
        )
        .is_err()
    );
    assert_eq!(actual, [19.0; 4]);
    assert!(
        value_tile(
            rows[..1].iter().map(<[u16; 4]>::as_slice),
            &[1.0; 4],
            &[1.0; 2],
            1,
            &mut actual,
        )
        .is_err()
    );
    assert_eq!(actual, [19.0; 4]);
    for bad in [f32::MAX, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        for bad_channel in 0..4 {
            let mut current = [1.0; 4];
            current[bad_channel] = bad;
            let mut actual = [19.0; 4];
            let mut expected = actual;
            assert!(value_tile([].into_iter(), &current, &[2.0], 0, &mut actual).is_err());
            assert!(
                value_tile_scalar([].into_iter(), &current, &[2.0], &(0..4), &mut expected).is_err()
            );
            assert_eq!(actual.map(f32::to_bits), expected.map(f32::to_bits));
            for (channel, &value) in actual.iter().enumerate() {
                assert_eq!(value, if channel < bad_channel { 2.0 } else { 19.0 });
            }
        }
    }
}
