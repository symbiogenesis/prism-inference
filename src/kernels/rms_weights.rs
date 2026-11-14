//! A finite immutable weight borrow shared by one recurrent layer traversal.

use crate::{Result, invalid};

/// Admits identical RMS weights once for all heads and tokens in a traversal.
///
/// The immutable borrow prevents replacement or mutation until every consumer
/// has joined. Input, epsilon, geometry and output checks remain per operation.
/// No transformed weights, cached pointers or persistent storage are retained.
#[derive(Clone, Copy)]
#[expect(
    clippy::redundant_pub_crate,
    reason = "The private module reexports this internal type crate-wide through kernels."
)]
pub(crate) struct RmsNormWeights<'a> {
    values: &'a [f32],
}

impl<'a> RmsNormWeights<'a> {
    pub(crate) fn new(values: &'a [f32]) -> Result<Self> {
        if !super::is_finite_slice(values) {
            return Err(invalid("nonfinite tensor or activation value"));
        }
        Ok(Self { values })
    }

    pub(crate) fn swiglu_in_place<const PINNED: bool>(
        self,
        values: &mut [f32],
        epsilon: f32,
        gate: &[f32],
    ) -> Result<()> {
        super::rms_norm_swiglu_in_place_inner::<PINNED, true>(values, self.values, epsilon, gate)
    }

    pub(crate) fn swiglu_into<const PINNED: bool>(
        self,
        input: &[f32],
        epsilon: f32,
        gate: &[f32],
        output: &mut [f32],
    ) -> Result<()> {
        super::rms_norm_swiglu_into_inner::<PINNED, true>(input, self.values, epsilon, gate, output)
    }
}

#[cfg(test)]
mod tests;
