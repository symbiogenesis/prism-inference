// Copyright (c) 2023-2026 The ggml/llama.cpp authors
// SPDX-License-Identifier: MIT
//! Rust-owned numerical comparisons from tests/test-backend-ops.cpp.

use std::collections::HashMap;
use std::hash::Hash;

/// Each product rounds to binary32 before binary64 accumulation, as in the source.
pub(super) fn nmse(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mut numerator = 0.0_f64;
    let mut denominator = 0.0_f64;
    for (&a, &b) in a.iter().zip(b) {
        let difference = a - b;
        numerator += f64::from(difference * difference);
        denominator += f64::from(a * a);
    }
    numerator / denominator
}

/// The upstream name denotes multiset distance: count differences / (2*n).
fn multiset_distance<T: Eq + Hash>(
    a: impl IntoIterator<Item = T>,
    b: impl IntoIterator<Item = T>,
    n: usize,
) -> f64 {
    let mut counts = HashMap::<T, (i64, i64)>::new();
    for value in a {
        counts.entry(value).or_default().0 += 1;
    }
    for value in b {
        counts.entry(value).or_default().1 += 1;
    }
    let difference: u64 = counts.values().map(|(a, b)| a.abs_diff(*b)).sum();
    difference as f64 / (2 * n) as f64
}

fn float_key(value: f32) -> u32 {
    // std::equal_to<float> regards the two zeros as equal. NaNs fail element checks first.
    if value == 0.0 { 0 } else { value.to_bits() }
}

pub(super) fn jdst(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    multiset_distance(
        a.iter().copied().map(float_key),
        b.iter().copied().map(float_key),
        a.len(),
    )
}

pub(super) fn mean_abs_asymm(a: &[f32], b: &[f32], expected: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mut sum = 0.0_f64;
    let mut count = 0_usize;
    for (&a, &b) in a.iter().zip(b) {
        if !expected.is_empty() && !expected.iter().any(|&value| (a - value).abs() < 1.0e-3_f32) {
            continue;
        }
        let asymmetry = (a - b) / (a + b);
        sum += f64::from(asymmetry.abs());
        count += 1;
    }
    sum / count as f64
}

pub(super) fn two_point(upper: f32, lower: f32, epsilon: f32) -> f32 {
    (upper - lower) / (2.0_f32 * epsilon)
}

pub(super) fn four_point(
    upper: f32,
    upper_half: f32,
    lower_half: f32,
    lower: f32,
    epsilon: f32,
) -> f32 {
    // Unsuffixed source constants promote the four-point expression to double.
    ((8.0 * f64::from(upper_half) + f64::from(lower)
        - (8.0 * f64::from(lower_half) + f64::from(upper)))
        / (6.0 * f64::from(epsilon))) as f32
}

pub(super) fn element_failure(a: &[f32], b: &[f32]) -> Option<(usize, &'static str)> {
    if a.len() != b.len() {
        return Some((0, "element count differs"));
    }
    for (index, (&a, &b)) in a.iter().zip(b).enumerate() {
        if a.is_nan() || b.is_nan() {
            return Some((index, "NaN element"));
        }
        let a_special = a.is_infinite() || a.abs() == f32::MAX;
        let b_special = b.is_infinite() || b.abs() == f32::MAX;
        if a_special != b_special {
            return Some((index, "infinity/FLT_MAX differs from finite element"));
        }
        if a_special && a.is_sign_negative() != b.is_sign_negative() {
            return Some((index, "infinity/FLT_MAX sign differs"));
        }
    }
    None
}

pub(super) fn get_rows_exact(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count() as f64
}

pub(super) fn maximum_absolute(a: &[f32], b: &[f32]) -> f64 {
    a.iter().chain(b).fold(0.0_f64, |maximum, &value| {
        maximum.max(f64::from(value.abs()))
    })
}

pub(super) fn sorted_nmse(a: &[f32], b: &[f32]) -> f64 {
    let mut a = a.to_vec();
    let mut b = b.to_vec();
    // NaNs have already failed the source element checks.
    a.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    b.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    nmse(&a, &b)
}

pub(super) fn top_k(
    a: &[f32],
    b: &[f32],
    ties: Option<(&[f32], usize, usize)>,
) -> Result<f64, String> {
    if a.len() != b.len() {
        return Err("top-k element count differs".into());
    }
    let indices = |values: &[f32]| -> Result<Vec<i32>, String> {
        values
            .iter()
            .map(|&value| {
                if !value.is_finite()
                    || f64::from(value) < f64::from(i32::MIN)
                    || f64::from(value) > f64::from(i32::MAX)
                {
                    Err("top-k output cannot be represented by the original int32 index".into())
                } else {
                    Ok(value as i32)
                }
            })
            .collect()
    };
    if let Some((source, columns, k)) = ties {
        if k == 0
            || columns == 0
            || source.len() % columns != 0
            || a.len() != (source.len() / columns) * k
        {
            return Err("top-k ties shape assertion failed".into());
        }
        let mut difference = 0.0_f64;
        for (row, (a, b)) in a.chunks_exact(k).zip(b.chunks_exact(k)).enumerate() {
            let mut ia = indices(a)?;
            let mut ib = indices(b)?;
            let lookup = |indices: &[i32]| -> Result<Vec<f32>, String> {
                indices
                    .iter()
                    .map(|&index| {
                        let index = usize::try_from(index)
                            .map_err(|_| "negative top-k index".to_owned())?;
                        if index >= columns {
                            return Err("top-k index outside input row".into());
                        }
                        Ok(source[row * columns + index])
                    })
                    .collect()
            };
            difference += jdst(&lookup(&ia)?, &lookup(&ib)?);
            ia.sort_unstable();
            ib.sort_unstable();
            if ia.windows(2).any(|pair| pair[0] == pair[1]) {
                difference += 1.0;
            }
            if ib.windows(2).any(|pair| pair[0] == pair[1]) {
                difference += 1.0;
            }
        }
        Ok(difference)
    } else {
        let ia = indices(a)?;
        let ib = indices(b)?;
        let mut difference = 0.0_f64;
        for ((&a, &b), (&ia, &ib)) in a.iter().zip(b).zip(ia.iter().zip(&ib)) {
            difference += f64::from((a - ia as f32).abs());
            difference += f64::from((b - ib as f32).abs());
        }
        Ok(difference + multiset_distance(ia, ib, a.len()))
    }
}

#[cfg(test)]
mod tests {
    use super::super::number::Number;
    use super::*;

    #[test]
    fn original_comparison_formula_guards() -> Result<(), String> {
        let (program, geometry) = super::super::program::load()?;
        assert_eq!(program.classes.len(), 114);
        assert_eq!(geometry.eval.len(), 22_817);
        assert_eq!(geometry.perf.len(), 1_530);
        assert!(nmse(&[0.0], &[0.0]).is_nan());
        assert!(mean_abs_asymm(&[0.25], &[0.5], &[1.0]).is_nan());
        assert_eq!(get_rows_exact(&[0.0], &[-0.0]), 1.0);
        assert_eq!(jdst(&[0.0, 1.0, 1.0], &[-0.0, 1.0, 2.0]), 1.0 / 3.0);
        assert!(element_failure(&[f32::MAX], &[f32::INFINITY]).is_none());
        assert!(element_failure(&[f32::MAX], &[f32::NEG_INFINITY]).is_some());
        assert!(element_failure(&[f32::NAN], &[f32::NAN]).is_some());
        assert_eq!(
            top_k(&[0.0, 2.0], &[1.0, 2.0], Some((&[1.0, 1.0, 2.0], 3, 2))),
            Ok(0.0)
        );
        assert_eq!(
            top_k(&[0.0, 0.0], &[1.0, 2.0], Some((&[1.0, 1.0, 1.0], 3, 2))),
            Ok(1.0)
        );
        assert_eq!(two_point(3.0, 1.0, 0.5), 2.0);
        assert_eq!(four_point(3.0, 2.5, 1.5, 1.0, 0.5), 2.0);
        assert!(Number::I32(-1).equal(Number::U32(u32::MAX)));
        assert!(Number::I64(-1).less(Number::U32(u32::MAX)));
        let mut jumped = super::super::random::CrtRandom::new(1);
        let mut sequential = jumped.clone();
        jumped.skip(257);
        for _ in 0..257 {
            let _ = sequential.next();
        }
        assert_eq!(jumped.next(), sequential.next());
        Ok(())
    }
}
