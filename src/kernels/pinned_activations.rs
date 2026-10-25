//! Safe scalar reproduction of pinned Windows ARM64 graph activation arithmetic.
//!
//! Reference: Prism `73bfe6cc485554822f80c473a3c0c4f25e80f41b`,
//! `ggml/src/ggml-cpu/vec.h` and `vec.cpp`. Upstream MIT terms are retained in
//! `PRISM-LICENSE.txt`. This numerical path makes no optimization claim.

#![allow(
    clippy::redundant_pub_crate,
    reason = "Keep graph-only activation helpers crate scoped while satisfying unreachable_pub."
)]

use crate::{Result, invalid};

// Reproduce the pinned ggml_v_expf NEON instructions lane by lane. Explicit
// mul_add retains their single rounding; bit arithmetic mirrors uint32 lanes.
// Scalar tails deliberately continue to call the ordinary exp implementation.
pub(super) fn vector_exp(value: f32) -> f32 {
    let z = value.mul_add(f32::from_bits(0x3fb8_aa3b), 12_582_912.0);
    let n = z - 12_582_912.0;
    let remainder = (-n).mul_add(f32::from_bits(0x3f31_7200), value);
    let remainder = (-n).mul_add(f32::from_bits(0x35bf_be8e), remainder);
    let exponent = z.to_bits().wrapping_shl(23);
    let k = f32::from_bits(exponent.wrapping_add(1.0_f32.to_bits()));
    let square = remainder * remainder;
    let inner = f32::from_bits(0x3c07_2010)
        .mul_add(remainder, f32::from_bits(0x3d2b_9f17))
        .mul_add(
            square,
            f32::from_bits(0x3e2a_af33).mul_add(remainder, f32::from_bits(0x3eff_fedb)),
        );
    let polynomial = inner.mul_add(square, f32::from_bits(0x3f7f_fff6) * remainder);
    if n.abs() > 126.0 {
        let shift = if n <= 0.0 { 0x8200_0000_u32 } else { 0 };
        let s1 = f32::from_bits(shift.wrapping_add(0x7f00_0000));
        if n.abs() > 192.0 {
            s1 * s1
        } else {
            let s2 = f32::from_bits(exponent.wrapping_sub(shift));
            s2.mul_add(polynomial, s2) * s1
        }
    } else {
        polynomial.mul_add(k, k)
    }
}

fn denominator(value: f32, vectorized: bool) -> f32 {
    1.0 + if vectorized {
        vector_exp(0.0 - value)
    } else {
        (-value).exp()
    }
}

/// Applies the pinned native `SiLU` row contract without allocating.
pub(crate) fn silu_in_place(values: &mut [f32]) {
    let vectorized = values.len() / 4 * 4;
    for (index, value) in values.iter_mut().enumerate() {
        *value /= denominator(*value, index < vectorized);
    }
}

/// Applies the pinned native `SwiGLU` row contract to the up projection.
///
/// # Errors
/// Rejects unequal row lengths before changing the up projection.
pub(crate) fn swiglu_in_place(gate: &[f32], up: &mut [f32]) -> Result<()> {
    if gate.len() != up.len() {
        return Err(invalid("pinned SwiGLU requires matching gate and up rows"));
    }
    let vectorized = gate.len() / 4 * 4;
    for (index, (&gate, up)) in gate.iter().zip(up).enumerate() {
        // The compiled pinned SwiGLU reassociates the numerator before division.
        *up = (gate * *up) / denominator(gate, index < vectorized);
    }
    Ok(())
}

/// Applies the pinned native `SwiGLU` row contract into existing output storage.
///
/// # Errors
/// Rejects unequal row lengths before changing the output.
pub(crate) fn swiglu_into(gate: &[f32], up: &[f32], output: &mut [f32]) -> Result<()> {
    if gate.len() != up.len() || gate.len() != output.len() {
        return Err(invalid(
            "pinned SwiGLU requires matching input and output rows",
        ));
    }
    let vectorized = gate.len() / 4 * 4;
    for (index, ((&gate, &up), output)) in gate.iter().zip(up).zip(output).enumerate() {
        *output = (gate * up) / denominator(gate, index < vectorized);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use serde::Deserialize;

    use super::{silu_in_place, swiglu_in_place, swiglu_into};

    #[derive(Deserialize)]
    struct Capture {
        runtime_commit: String,
        ggml_cpu_sha256: String,
        cases: Vec<Case>,
    }

    #[derive(Deserialize)]
    struct Case {
        name: String,
        input_bits: Vec<u32>,
        weight_bits: Vec<u32>,
        silu_bits: Vec<u32>,
        swiglu_bits: Vec<u32>,
    }

    fn assert_native(actual: &[f32], expected: &[u32], case: &str) {
        assert_eq!(actual.len(), expected.len(), "{case}");
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            if f32::from_bits(expected).is_nan() {
                assert!(actual.is_nan(), "{case}, element {index}");
            } else {
                assert_eq!(actual.to_bits(), expected, "{case}, element {index}");
            }
        }
    }

    #[test]
    fn pinned_activations_match_native_random_tails_and_finite_edges() -> crate::Result<()> {
        let capture: Capture = serde_json::from_str(include_str!(
            "../../tests/fixtures/swift-activation-parity.json"
        ))?;
        assert_eq!(
            capture.runtime_commit,
            "73bfe6cc485554822f80c473a3c0c4f25e80f41b"
        );
        assert_eq!(
            capture.ggml_cpu_sha256,
            "701b354a38b25e5c6dc57a1559f71cd26b2d0f8d69c1576a0e77a59a4273d2ab"
        );
        for case in capture.cases {
            let gate = case
                .input_bits
                .into_iter()
                .map(f32::from_bits)
                .collect::<Vec<_>>();
            let up = case
                .weight_bits
                .into_iter()
                .map(f32::from_bits)
                .collect::<Vec<_>>();
            let mut unary = gate.clone();
            silu_in_place(&mut unary);
            assert_native(&unary, &case.silu_bits, &case.name);
            let mut inplace = up.clone();
            swiglu_in_place(&gate, &mut inplace)?;
            assert_native(&inplace, &case.swiglu_bits, &case.name);
            let mut output = vec![99.0; up.len()];
            swiglu_into(&gate, &up, &mut output)?;
            assert_native(&output, &case.swiglu_bits, &case.name);
        }
        Ok(())
    }

    #[test]
    fn pinned_swiglu_rejects_mismatched_rows_before_mutation() {
        let mut output = [99.0];
        assert!(swiglu_in_place(&[], &mut output).is_err());
        assert_eq!(output[0].to_bits(), 99.0_f32.to_bits());
        assert!(swiglu_into(&[1.0], &[], &mut output).is_err());
        assert_eq!(output[0].to_bits(), 99.0_f32.to_bits());
    }
}
