//! Explicit sequential CPU observation and additive FFN patches for research.
//!
//! This API is separate from serving. A probe consumes an empty scalar graph,
//! seals immutable factors into a fresh checkpoint identity, and exposes no
//! tiled-prefill, accelerator, adapter-scale or factor mutation methods.

use std::fmt::Write as _;
use std::sync::Arc;

use bonsai_native_hash::Hasher;

use super::{MatvecBackend, Qwen35, Qwen35Checkpoint};
use crate::{Decoder, Result, invalid};

/// Numerical profile for the experimental original-basis FFN observation path.
pub const EXECUTION_PROFILE: &str = "experimental-ffn-cpu-scalar-unfused-v1";
/// Basis of the post-SwiGLU FFN activation supplied to a patch.
pub const INPUT_BASIS: &str = "original-ffn-hidden";

/// Borrowed values at one selected FFN site during a provisional token.
///
/// `activation` is the original, unrotated post-gating FFN hidden vector.
/// `residual` is the residual stream after the complete FFN, including the
/// existing adapter and any independent patch. These are different sites.
/// The callback precedes the token's state commit; consumers must discard its
/// staged observation if the surrounding evaluation subsequently fails.
pub struct FfnObservation<'a> {
    /// Zero-based decoder block index.
    pub layer: usize,
    /// Zero-based input token position in the current sequence.
    pub token_index: usize,
    /// Original post-gating FFN input to `ffn_down`, before Hadamard rotation.
    pub activation: &'a [f32],
    /// Residual stream after attention and the selected complete FFN.
    pub residual: &'a [f32],
}

/// Immutable rank-one correction applied after an existing adapter branch.
///
/// The scalar source-order operation is `output += scale * b * dot(a, h)`.
/// It never merges into, rewrites, or requantizes model or adapter tensors.
pub struct FfnPatch {
    a: Vec<f32>,
    b: Vec<f32>,
    scale: f32,
    identity: String,
}

impl FfnPatch {
    /// Validates finite factors and computes their exact binary32 identity.
    ///
    /// # Errors
    /// Rejects empty, oversized or nonfinite factors, nonfinite scale and
    /// native hashing/allocation failures. Site dimensions are checked again
    /// when consuming the graph into [`FfnProbe`].
    pub fn new(a: Vec<f32>, b: Vec<f32>, scale: f32) -> Result<Self> {
        if a.is_empty()
            || b.is_empty()
            || a.len() > super::MAX_WIDTH
            || b.len() > super::MAX_WIDTH
            || a.iter().chain(&b).any(|value| !value.is_finite())
            || !scale.is_finite()
        {
            return Err(invalid(
                "FFN patch requires bounded finite factors and scale",
            ));
        }
        let mut hash = Hasher::new()?;
        hash.update(b"ffn-rank-one-patch-f32-source-v1")?;
        hash.update(&u64::try_from(a.len())?.to_le_bytes())?;
        hash.update(&u64::try_from(b.len())?.to_le_bytes())?;
        hash.update(&scale.to_bits().to_le_bytes())?;
        for value in a.iter().chain(&b) {
            hash.update(&value.to_bits().to_le_bytes())?;
        }
        let mut identity = String::new();
        identity.try_reserve_exact(64)?;
        for byte in hash.finish()? {
            write!(identity, "{byte:02x}")?;
        }
        Ok(Self {
            a,
            b,
            scale,
            identity,
        })
    }

    /// Identity of exact factor shapes, bits, scale and arithmetic contract.
    #[must_use]
    pub fn identity(&self) -> &str {
        &self.identity
    }

    #[expect(
        clippy::suboptimal_flops,
        reason = "The declared source-order F32 contract rounds each product before reduction, scale and base addition; FMA changes patch arithmetic."
    )]
    pub(super) fn apply(&self, input: &[f32], output: &mut [f32]) -> Result<()> {
        if input.len() != self.a.len() || output.len() != self.b.len() {
            return Err(invalid(
                "FFN patch activation/output dimensions do not match",
            ));
        }
        // Preserve all base/adapter output bits, including signed zero, when
        // explicitly disabled. Do not even reduce the input in that case.
        if self.scale == 0.0 {
            return Ok(());
        }
        let mut dot = 0.0_f32;
        for (&factor, &value) in self.a.iter().zip(input) {
            dot += factor * value;
        }
        let scaled = dot * self.scale;
        if !scaled.is_finite()
            || output
                .iter()
                .zip(&self.b)
                .any(|(&base, &factor)| !(base + scaled * factor).is_finite())
        {
            return Err(invalid("FFN patch produced a nonfinite correction"));
        }
        for (base, &factor) in output.iter_mut().zip(&self.b) {
            *base += scaled * factor;
        }
        Ok(())
    }
}

pub(super) struct Intervention<'a> {
    pub(super) layer: usize,
    pub(super) patch: Option<&'a FfnPatch>,
    pub(super) observer: &'a mut dyn FnMut(FfnObservation<'_>) -> Result<()>,
}

/// A sealed experimental graph with one observed and optionally patched FFN.
///
/// Both patched and unpatched probes use the same separate original-basis FFN
/// operations at the selected site. This is an explicitly different execution
/// identity from the ordinary fused graph. Scale zero exactly bypasses patch
/// arithmetic within this profile; that does not claim parity with every fused
/// production CPU or accelerator path.
pub struct FfnProbe {
    graph: Qwen35,
    layer: usize,
    patch: Option<FfnPatch>,
}

impl FfnProbe {
    /// Consumes an empty scalar, one-worker CPU graph into a fixed FFN profile.
    ///
    /// Existing adapter factors and scale are preserved. All old checkpoints
    /// become foreign, even without a patch. A fresh probe is required to change
    /// factors, scale or site, preventing prefix reuse across those changes.
    ///
    /// # Errors
    /// Rejects nonempty state, non-scalar dispatch, multiple row workers,
    /// installed tiled-prefill/accelerator paths, invalid sites or mismatched
    /// factor dimensions, and identity allocation failures.
    pub fn new(mut graph: Qwen35, layer: usize, patch: Option<FfnPatch>) -> Result<Self> {
        if graph.position != 0
            || graph.requested_backend() != MatvecBackend::Scalar
            || graph.row_workers() != 1
            || graph.tiled_prefill.is_some()
            || graph.projection_executor.is_some()
        {
            return Err(invalid(
                "FFN probe requires an empty one-worker scalar CPU graph",
            ));
        }
        let selected = graph
            .layers
            .get(layer)
            .ok_or_else(|| invalid("FFN probe layer is outside the graph"))?;
        if selected.down.dimensions.len() != 2
            || patch
                .as_ref()
                .is_some_and(|patch| selected.down.dimensions != [patch.a.len(), patch.b.len()])
        {
            return Err(invalid(
                "FFN patch factors do not match selected down projection",
            ));
        }
        write!(
            graph.identity,
            ":{EXECUTION_PROFILE}:layer={layer}:patch={}",
            patch.as_ref().map_or("none", FfnPatch::identity),
        )?;
        graph.provenance = Arc::new(());
        // Capture must observe the final FFN even for state-only ingestion.
        graph.narrow_last_layer = false;
        Ok(Self {
            graph,
            layer,
            patch,
        })
    }

    /// Evaluates one token and borrows the selected activation and residual.
    ///
    /// The callback may inspect or copy bounded selected values. It cannot alter
    /// them. A callback error aborts before state commit; observations remain
    /// provisional until this entire method succeeds.
    ///
    /// # Errors
    /// Returns graph/callback errors and empties `logits` on failure.
    pub fn evaluate_observed_into(
        &mut self,
        token: u32,
        logits: &mut Vec<f32>,
        observer: &mut impl FnMut(FfnObservation<'_>) -> Result<()>,
    ) -> Result<()> {
        let mut intervention = Intervention {
            layer: self.layer,
            patch: self.patch.as_ref(),
            observer,
        };
        let result = self.graph.evaluate_token_with_executor(
            token,
            Some(logits),
            None,
            Some(&mut intervention),
        );
        if result.is_err() {
            logits.clear();
        }
        result
    }
}

impl Decoder for FfnProbe {
    type Checkpoint = Qwen35Checkpoint;

    fn identity(&self) -> &str {
        self.graph.identity()
    }

    fn vocab_size(&self) -> usize {
        self.graph.vocab_size()
    }

    fn checkpoint_bytes(&self) -> usize {
        self.graph.checkpoint_bytes()
    }

    fn reset(&mut self) -> Result<()> {
        self.graph.reset()
    }

    fn evaluate(&mut self, token: u32) -> Result<Vec<f32>> {
        let mut logits = Vec::new();
        self.evaluate_into(token, &mut logits)?;
        Ok(logits)
    }

    fn evaluate_into(&mut self, token: u32, logits: &mut Vec<f32>) -> Result<()> {
        self.evaluate_observed_into(token, logits, &mut |_| Ok(()))
    }

    fn ingest(&mut self, token: u32) -> Result<()> {
        let mut intervention = Intervention {
            layer: self.layer,
            patch: self.patch.as_ref(),
            observer: &mut |_| Ok(()),
        };
        self.graph
            .evaluate_token_with_executor(token, None, None, Some(&mut intervention))
    }

    fn checkpoint(&self) -> Result<Self::Checkpoint> {
        self.graph.checkpoint()
    }

    fn restore(&mut self, checkpoint: &Self::Checkpoint) -> Result<()> {
        self.graph.restore(checkpoint)
    }
}

#[cfg(test)]
mod tests {
    use super::FfnPatch;
    use crate::Result;

    #[test]
    fn rank_one_patch_adds_to_existing_output_in_declared_order() -> Result<()> {
        let patch = FfnPatch::new(vec![2.0, -1.0], vec![0.5, -0.25], 0.5)?;
        let mut output = [10.0, -4.0];
        patch.apply(&[3.0, 2.0], &mut output)?;
        assert_eq!(output, [11.0, -4.5]);
        Ok(())
    }

    #[test]
    fn rank_one_patch_zero_bypasses_and_failure_preserves_output() -> Result<()> {
        let zero = FfnPatch::new(vec![f32::MAX], vec![1.0], -0.0)?;
        let mut output = [-0.0_f32];
        zero.apply(&[f32::MAX], &mut output)?;
        assert_eq!(output[0].to_bits(), (-0.0_f32).to_bits());
        let patch = FfnPatch::new(vec![f32::MAX], vec![1.0], 1.0)?;
        assert!(patch.apply(&[f32::MAX], &mut output).is_err());
        assert_eq!(output[0].to_bits(), (-0.0_f32).to_bits());
        assert!(FfnPatch::new(vec![f32::NAN], vec![1.0], 1.0).is_err());
        assert!(FfnPatch::new(vec![1.0], vec![f32::INFINITY], 0.0).is_err());
        Ok(())
    }
}
