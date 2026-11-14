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
/// Original normalized residual basis used by an entire-FFN surrogate.
pub const SURROGATE_INPUT_BASIS: &str = "original-ffn-normalized-input";
/// Complete FFN branch, including the frozen existing adapter, before residual add.
pub const SURROGATE_TARGET_BASIS: &str = "complete-ffn-branch-delta";
/// Separate scalar graph identity for complete-group FFN omission research.
pub const GROUP_OMISSION_PROFILE: &str = "experimental-ffn-group-omission-cpu-scalar-unfused-v1";

/// Paired complete and retained branches evaluated on the same student input.
///
/// Both branches include the frozen standard adapter. The retained branch uses
/// zero for omitted original hidden groups before both the base down projection
/// and the adapter. The returned input is before any Hadamard rotation.
pub struct FfnGroupObservation {
    /// Original normalized pre-FFN residual.
    pub input: Vec<f32>,
    /// Complete FFN delta on this same student input, before residual addition.
    pub teacher_target: Vec<f32>,
    /// Group-omitted FFN delta actually propagated through the student graph.
    pub retained_branch: Vec<f32>,
}

pub(super) struct GroupOmission {
    groups: Vec<usize>,
    full_signs_sha256: String,
}

impl GroupOmission {
    pub(super) fn mask_hidden(&self, hidden: &mut [f32]) -> Result<()> {
        if hidden.len() != 17_408 {
            return Err(invalid(
                "group omission requires Swift original FFN hidden width",
            ));
        }
        for &group in &self.groups {
            hidden[group * 1024..(group + 1) * 1024].fill(0.0);
        }
        Ok(())
    }
}

/// Owned teacher observation returned only after the complete token succeeds.
///
/// Unlike [`FfnObservation`], these vectors describe the input and output of
/// the entire FFN branch. The target is captured before residual addition;
/// subtracting rounded residual streams cannot reconstruct it exactly.
pub struct FfnSurrogateObservation {
    /// Zero-based decoder block index.
    pub layer: usize,
    /// Zero-based input token position in the current sequence.
    pub token_index: usize,
    /// Original normalized pre-FFN residual, before Hadamard rotation.
    pub input: Vec<f32>,
    /// Complete FFN delta, including the existing adapter at its frozen scale.
    pub target: Vec<f32>,
}

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

type SurrogateObserver<'a> = &'a mut dyn FnMut(&[f32], &[f32]) -> Result<()>;
type GroupObserver<'a> = &'a mut dyn FnMut(&[f32], &[f32], &[f32]) -> Result<()>;

pub(super) struct Intervention<'a> {
    pub(super) layer: usize,
    pub(super) patch: Option<&'a FfnPatch>,
    pub(super) observer: &'a mut dyn FnMut(FfnObservation<'_>) -> Result<()>,
    pub(super) surrogate_observer: Option<SurrogateObserver<'a>>,
    pub(super) omission: Option<&'a GroupOmission>,
    pub(super) group_observer: Option<GroupObserver<'a>>,
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
    omission: Option<GroupOmission>,
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
            omission: None,
        })
    }

    /// Consumes an unpatched teacher into a reset, fixed one-site omission graph.
    ///
    /// Complete original hidden groups are zeroed only after computing the full
    /// teacher branch. The retained base down projection and standard adapter
    /// receive the same zeroed original hidden input. All weights remain loaded
    /// and the full gate/up computation still executes; this is capture, not an
    /// accelerated or memory-saving serving implementation.
    ///
    /// # Errors
    /// Rejects patched/already-omitted graphs, noncanonical group selection,
    /// unsupported Swift dimensions/transforms/signs, hashing and reset failures.
    pub fn with_group_omission(mut self, groups: Vec<usize>) -> Result<Self> {
        if self.patch.is_some()
            || self.omission.is_some()
            || groups.is_empty()
            || groups.len() >= 17
            || groups.iter().any(|&group| group >= 17)
            || groups.windows(2).any(|pair| pair[0] >= pair[1])
        {
            return Err(invalid(
                "group omission needs sorted distinct groups 0..17 and an unpatched teacher",
            ));
        }
        let selected = &self.graph.layers[self.layer];
        if selected.down.dimensions != [17_408, 5_120]
            || selected.up.dimensions != [5_120, 17_408]
            || selected.gate.dimensions != [5_120, 17_408]
            || self.graph.transforms.block != 1024
            || self.graph.transforms.basis(&selected.down) != (true, false)
        {
            return Err(invalid(
                "group omission requires the pinned Swift block-1024 FFN transform",
            ));
        }
        let mut hash = Hasher::new()?;
        // Swift's metadata order is [5120,6144,17408]. Hash canonical I32-LE
        // +/-1 values, matching the complete sign_values evidence contract.
        for width in [5_120, 6_144, 17_408] {
            let signs = self
                .graph
                .transforms
                .signs
                .get(&width)
                .filter(|signs| signs.len() == width)
                .ok_or_else(|| invalid("group omission requires complete explicit Swift signs"))?;
            for &sign in signs {
                let value: i32 = if sign.to_bits() == 1.0_f32.to_bits() {
                    1
                } else if sign.to_bits() == (-1.0_f32).to_bits() {
                    -1
                } else {
                    return Err(invalid("group omission signs must be exact +/-1"));
                };
                hash.update(&value.to_le_bytes())?;
            }
        }
        let mut full_signs_sha256 = String::new();
        full_signs_sha256.try_reserve_exact(64)?;
        for byte in hash.finish()? {
            write!(full_signs_sha256, "{byte:02x}")?;
        }
        self.graph.reset()?;
        write!(
            self.graph.identity,
            ":{GROUP_OMISSION_PROFILE}:groups={groups:?}:signs={full_signs_sha256}"
        )?;
        self.graph.provenance = Arc::new(());
        self.omission = Some(GroupOmission {
            groups,
            full_signs_sha256,
        });
        Ok(self)
    }

    /// Canonical complete I32-LE metadata sign digest for an omission probe.
    #[must_use]
    pub fn omission_signs_sha256(&self) -> Option<&str> {
        self.omission
            .as_ref()
            .map(|omission| omission.full_signs_sha256.as_str())
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
            surrogate_observer: None,
            omission: self.omission.as_ref(),
            group_observer: None,
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

    /// Evaluates one token and returns an entire-FFN teacher pair on success.
    ///
    /// Original normalized input and the complete adapter-inclusive FFN delta
    /// are copied before residual addition. Copies remain provisional and are
    /// dropped if any later layer, output projection or state staging fails.
    /// Existing adapter factors and scale remain fixed by this probe.
    ///
    /// # Errors
    /// Rejects independently patched probes and graph/allocation failures.
    /// Errors clear `logits`, return no observation and commit no token state.
    pub fn evaluate_surrogate_into(
        &mut self,
        token: u32,
        logits: &mut Vec<f32>,
    ) -> Result<FfnSurrogateObservation> {
        logits.clear();
        if self.patch.is_some() || self.omission.is_some() {
            return Err(invalid("FFN surrogate teacher rejects independent patches"));
        }
        let layer = self.layer;
        let token_index = self.graph.position;
        let mut observation = None;
        let mut capture = |input: &[f32], target: &[f32]| {
            observation = Some(FfnSurrogateObservation {
                layer,
                token_index,
                input: super::copy_values(input)?,
                target: super::copy_values(target)?,
            });
            Ok(())
        };
        let mut intervention = Intervention {
            layer,
            patch: None,
            observer: &mut |_| Ok(()),
            surrogate_observer: Some(&mut capture),
            omission: None,
            group_observer: None,
        };
        let result = self.graph.evaluate_token_with_executor(
            token,
            Some(logits),
            None,
            Some(&mut intervention),
        );
        if let Err(error) = result {
            logits.clear();
            return Err(error);
        }
        observation.ok_or_else(|| invalid("selected FFN surrogate observation is absent"))
    }

    /// Evaluates and returns paired branches only after the entire token commits.
    ///
    /// # Errors
    /// Rejects a teacher-only probe or later graph/copy failures. On failure no
    /// observation is returned, logits are cleared and no token state commits.
    pub fn evaluate_group_omission_into(
        &mut self,
        token: u32,
        logits: &mut Vec<f32>,
    ) -> Result<FfnGroupObservation> {
        logits.clear();
        let omission = self
            .omission
            .as_ref()
            .ok_or_else(|| invalid("paired branches require a group-omission probe"))?;
        let mut observation = None;
        let mut capture = |input: &[f32], teacher: &[f32], retained: &[f32]| {
            observation = Some(FfnGroupObservation {
                input: super::copy_values(input)?,
                teacher_target: super::copy_values(teacher)?,
                retained_branch: super::copy_values(retained)?,
            });
            Ok(())
        };
        let mut intervention = Intervention {
            layer: self.layer,
            patch: None,
            observer: &mut |_| Ok(()),
            surrogate_observer: None,
            omission: Some(omission),
            group_observer: Some(&mut capture),
        };
        let result = self.graph.evaluate_token_with_executor(
            token,
            Some(logits),
            None,
            Some(&mut intervention),
        );
        if let Err(error) = result {
            logits.clear();
            return Err(error);
        }
        observation.ok_or_else(|| invalid("selected group-omission observation is absent"))
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
            surrogate_observer: None,
            omission: self.omission.as_ref(),
            group_observer: None,
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
    use super::{FfnPatch, GroupOmission};
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

    #[test]
    fn group_omission_preserves_transform_blocks_and_original_adapter_coordinates() -> Result<()> {
        let omission = GroupOmission {
            groups: vec![0, 8, 16],
            full_signs_sha256: String::new(),
        };
        let mut hidden = vec![1.0_f32; 17_408];
        // Distinct original positions make an accidental compact/reindexed
        // standard-adapter input observably wrong at group boundaries.
        for group in 0..17 {
            hidden[group * 1024] = 2.0;
            hidden[(group + 1) * 1024 - 1] = -1.0;
        }
        let original = hidden.clone();
        omission.mask_hidden(&mut hidden)?;
        for group in 0..17 {
            let range = group * 1024..(group + 1) * 1024;
            if [0, 8, 16].contains(&group) {
                assert!(hidden[range].iter().all(|value| value.to_bits() == 0));
            } else {
                assert_eq!(hidden[range.clone()], original[range]);
            }
        }
        let mut transformed = hidden.clone();
        for (group, block) in transformed.as_chunks_mut::<1024>().0.iter_mut().enumerate() {
            // Retained signs use each original group's slice. Whole-block
            // omission must commute with this independent signed transform.
            let sign = if group % 2 == 0 { -1.0 } else { 1.0 };
            for value in block.iter_mut() {
                *value *= sign;
            }
            crate::kernels::hadamard(block)?;
            if [0, 8, 16].contains(&group) {
                assert!(block.iter().all(|value| *value == 0.0));
            } else {
                let mut expected = original[group * 1024..(group + 1) * 1024].to_vec();
                for value in &mut expected {
                    *value *= sign;
                }
                crate::kernels::hadamard(&mut expected)?;
                assert_eq!(&*block, expected.as_slice());
            }
        }
        // A standard adapter reading the final original coordinate must see
        // zero, while the last retained coordinate retains its original value.
        assert_eq!(hidden[17_407], 0.0);
        assert_eq!(hidden[16_383], -1.0);
        assert!(omission.mask_hidden(&mut [1.0; 1024]).is_err());
        Ok(())
    }
}
