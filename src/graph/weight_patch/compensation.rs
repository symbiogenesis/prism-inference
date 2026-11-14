//! Immutable, ordered binary32 recovery applied to the normalized FFN input.

use std::fmt::Write as _;

use bonsai_native_hash::Hasher;

use crate::{Result, invalid};

/// Application order shared with Runner's exported-factor development screen.
pub const FACTOR_APPLICATION_ARITHMETIC: &str = "rank-major-source-order-f32-dot-add-v1";
/// Separate graph profile for retained FFN branches with frozen recovery factors.
pub const GROUP_COMPENSATION_PROFILE: &str =
    "experimental-ffn-group-compensation-cpu-scalar-unfused-v1";

/// Frozen low-rank recovery in the original normalized pre-FFN input basis.
///
/// Each rank computes an ordered binary32 dot, then adds `dot * b[k][j]`
/// directly to the retained branch in increasing rank order. The standard
/// adapter is already in that branch and is never applied again.
pub struct FfnCompensation {
    a: Vec<Vec<f32>>,
    b: Vec<Vec<f32>>,
    identity: String,
}

impl FfnCompensation {
    /// Checks bounded finite rank-major factors and binds exact bits and order.
    ///
    /// # Errors
    /// Rejects rank outside 1..=16, empty/nonuniform/oversized factor widths,
    /// nonfinite factors and hash/allocation failures. The probe additionally
    /// checks the selected site's actual input/output widths.
    pub fn new(a: Vec<Vec<f32>>, b: Vec<Vec<f32>>) -> Result<Self> {
        if a.is_empty()
            || a.len() > 16
            || a.len() != b.len()
            || a[0].is_empty()
            || b[0].is_empty()
            || a[0].len() > super::super::MAX_WIDTH
            || b[0].len() > super::super::MAX_WIDTH
            || a.iter().any(|row| row.len() != a[0].len())
            || b.iter().any(|row| row.len() != b[0].len())
            || a.iter().chain(&b).flatten().any(|value| !value.is_finite())
        {
            return Err(invalid("invalid normalized-input FFN compensation factors"));
        }
        let mut hash = Hasher::new()?;
        hash.update(b"prism-ffn-group-compensation-f32-v1\0")?;
        hash.update(FACTOR_APPLICATION_ARITHMETIC.as_bytes())?;
        for width in [a.len(), a[0].len(), b[0].len()] {
            hash.update(&u64::try_from(width)?.to_le_bytes())?;
        }
        for value in a.iter().chain(&b).flatten() {
            hash.update(&value.to_le_bytes())?;
        }
        let mut identity = String::new();
        identity.try_reserve_exact(64)?;
        for byte in hash.finish()? {
            write!(identity, "{byte:02x}")?;
        }
        Ok(Self { a, b, identity })
    }

    /// Exact factors, shape and arithmetic identity.
    #[must_use]
    pub fn identity(&self) -> &str {
        &self.identity
    }

    pub(super) fn dimensions(&self) -> (usize, usize) {
        (self.a[0].len(), self.b[0].len())
    }

    #[expect(
        clippy::suboptimal_flops,
        reason = "The published arithmetic rounds each product and addition separately; FMA or a final summed correction changes the function."
    )]
    pub(in crate::graph) fn apply(&self, input: &[f32], output: &mut [f32]) -> Result<()> {
        if self.dimensions() != (input.len(), output.len())
            || input
                .iter()
                .chain(output.iter())
                .any(|value| !value.is_finite())
        {
            return Err(invalid(
                "FFN compensation input/output dimensions or values differ",
            ));
        }
        // Keep the retained branch unchanged if any later rank overflows.
        let mut staged = super::super::copy_values(output)?;
        for (a, b) in self.a.iter().zip(&self.b) {
            let mut dot = 0.0_f32;
            for (&factor, &value) in a.iter().zip(input) {
                dot += factor * value;
            }
            if !dot.is_finite() {
                return Err(invalid("FFN compensation produced a nonfinite dot"));
            }
            for (value, &factor) in staged.iter_mut().zip(b) {
                *value += dot * factor;
                if !value.is_finite() {
                    return Err(invalid("FFN compensation produced a nonfinite branch"));
                }
            }
        }
        output.copy_from_slice(&staged);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::FfnCompensation;
    use crate::Result;

    fn tiny_probe() -> Result<super::super::FfnProbe> {
        use crate::{
            graph::Qwen35,
            kernels::MatvecBackend,
            model::{Model, Tensor},
        };
        use bonsai_gguf::MetadataValue;
        use std::{collections::BTreeMap, sync::Arc};
        let mut metadata = BTreeMap::new();
        metadata.insert("general.architecture".into(), MetadataValue::from("qwen35"));
        for (suffix, value) in [
            ("block_count", 1_u32),
            ("context_length", 16),
            ("embedding_length", 2),
            ("feed_forward_length", 2),
            ("attention.head_count", 1),
            ("attention.head_count_kv", 1),
            ("attention.key_length", 2),
            ("attention.value_length", 2),
            ("rope.dimension_count", 2),
            ("ssm.state_size", 2),
            ("ssm.group_count", 1),
            ("ssm.time_step_rank", 1),
            ("ssm.inner_size", 2),
            ("ssm.conv_kernel", 2),
            ("full_attention_interval", 1),
        ] {
            metadata.insert(format!("qwen35.{suffix}"), MetadataValue::from(value));
        }
        metadata.insert(
            "qwen35.attention.layer_norm_rms_epsilon".into(),
            MetadataValue::from(0.000_001_f32),
        );
        metadata.insert(
            "qwen35.rope.freq_base".into(),
            MetadataValue::from(10_000.0_f32),
        );
        metadata.insert(
            "qwen35.rope.dimension_sections".into(),
            MetadataValue::from(vec![1_u32, 0, 0, 0]),
        );
        metadata.insert(
            "qwen35.attention.recurrent_layers".into(),
            MetadataValue::from(vec![false]),
        );
        let mut tensors = BTreeMap::new();
        for (name, shape, values) in [
            (
                "token_embd.weight",
                vec![2, 2],
                vec![1.0_f32, 0.0, 0.0, 1.0],
            ),
            ("output.weight", vec![2, 2], vec![1.0, 0.0, 0.0, 1.0]),
            ("output_norm.weight", vec![2], vec![1.0; 2]),
            ("blk.0.attn_norm.weight", vec![2], vec![1.0; 2]),
            ("blk.0.post_attention_norm.weight", vec![2], vec![1.0; 2]),
            ("blk.0.ffn_up.weight", vec![2, 2], vec![0.0; 4]),
            ("blk.0.ffn_gate.weight", vec![2, 2], vec![0.0; 4]),
            ("blk.0.ffn_down.weight", vec![2, 2], vec![0.0; 4]),
            ("blk.0.attn_q.weight", vec![2, 4], vec![0.0; 8]),
            ("blk.0.attn_k.weight", vec![2, 2], vec![0.0; 4]),
            ("blk.0.attn_v.weight", vec![2, 2], vec![1.0, 0.0, 0.0, 1.0]),
            ("blk.0.attn_output.weight", vec![2, 2], vec![0.0; 4]),
            ("blk.0.attn_q_norm.weight", vec![2], vec![1.0; 2]),
            ("blk.0.attn_k_norm.weight", vec![2], vec![1.0; 2]),
        ] {
            tensors.insert(
                name.into(),
                Tensor {
                    name: name.into(),
                    dimensions: shape,
                    kind: 0,
                    data: Arc::from(
                        values
                            .iter()
                            .flat_map(|value| value.to_le_bytes())
                            .collect::<Vec<_>>(),
                    ),
                },
            );
        }
        super::super::FfnProbe::new(
            Qwen35::with_backend(
                Model {
                    metadata,
                    tensors,
                    identity: "tiny-ffn-recovery-fixture".into(),
                },
                8,
                MatvecBackend::Scalar,
            )?,
            0,
            None,
        )
    }

    #[test]
    fn compensation_graph_seam_preserves_state_identity_ingestion_and_failure_atomicity()
    -> Result<()> {
        use crate::Decoder;
        let factors = || FfnCompensation::new(vec![vec![1.0, 0.0]], vec![vec![0.0, 0.5]]);
        let original = tiny_probe()?;
        let old_checkpoint = original.checkpoint()?;
        assert!(tiny_probe()?.with_group_compensation(factors()?).is_err());
        // The private installer exercises the actual graph seam on a tiny
        // shape; public omission admission continues to require pinned Swift.
        let mut recovered = original.install_compensation(factors()?)?;
        assert!(recovered.restore(&old_checkpoint).is_err());
        let mut plain = tiny_probe()?;
        let plain_logits = plain.evaluate(0)?;
        let mut observed_residual = Vec::new();
        let recovered_logits = {
            let mut logits = Vec::new();
            recovered.evaluate_observed_into(0, &mut logits, &mut |row| {
                // The FFN hidden vector is all zero: a nonzero correction
                // proves the recovery reads normalized input, not that hidden.
                assert!(row.activation.iter().all(|value| *value == 0.0));
                observed_residual.extend_from_slice(row.residual);
                Ok(())
            })?;
            logits
        };
        let expected_normalized_first = 1.0_f32 / (0.5_f32 + 0.000_001).sqrt();
        assert_eq!(observed_residual[0].to_bits(), 1.0_f32.to_bits());
        assert_eq!(
            observed_residual[1].to_bits(),
            (0.5 * expected_normalized_first).to_bits()
        );
        assert_ne!(recovered_logits, plain_logits);
        let checkpoint = recovered.checkpoint()?;
        let next = recovered.evaluate(1)?;
        recovered.restore(&checkpoint)?;
        assert_eq!(next, recovered.evaluate(1)?);
        let mut ingested = tiny_probe()?.install_compensation(factors()?)?;
        ingested.ingest(0)?;
        assert_eq!(next, ingested.evaluate(1)?);
        let mut overflow = tiny_probe()?.install_compensation(FfnCompensation::new(
            vec![vec![f32::MAX, 0.0]],
            vec![vec![1.0; 2]],
        )?)?;
        let mut logits = vec![1.0];
        assert!(overflow.evaluate_into(0, &mut logits).is_err());
        assert_eq!(logits, Vec::<f32>::new());
        let expected = tiny_probe()?
            .install_compensation(FfnCompensation::new(
                vec![vec![f32::MAX, 0.0]],
                vec![vec![1.0; 2]],
            )?)?
            .evaluate(1)?;
        assert_eq!(overflow.evaluate(1)?, expected);
        Ok(())
    }

    #[test]
    fn compensation_uses_f32_dot_and_rank_order_and_stages_failures() -> Result<()> {
        let factors = FfnCompensation::new(
            vec![vec![16_777_216.0, 1.0, -16_777_216.0]],
            vec![vec![1.0]],
        )?;
        let mut branch = [2.0];
        factors.apply(&[1.0; 3], &mut branch)?;
        assert_eq!(branch, [2.0]);
        let factors = FfnCompensation::new(
            vec![vec![1.0], vec![-16_777_216.0]],
            vec![vec![1.0], vec![1.0]],
        )?;
        let mut branch = [16_777_216.0];
        factors.apply(&[1.0], &mut branch)?;
        assert_eq!(branch, [0.0]);
        let factors = FfnCompensation::new(
            vec![vec![1.0], vec![f32::MAX]],
            vec![vec![1.0], vec![f32::MAX]],
        )?;
        let mut branch = [3.0];
        assert!(factors.apply(&[1.0], &mut branch).is_err());
        assert_eq!(branch, [3.0]);
        assert!(factors.apply(&[2.0], &mut branch).is_err());
        assert_eq!(branch, [3.0]);
        Ok(())
    }

    #[test]
    fn compensation_binds_exact_bits_shapes_and_rejects_invalid_factors() -> Result<()> {
        let first = FfnCompensation::new(vec![vec![1.0]], vec![vec![0.0]])?;
        let changed = FfnCompensation::new(vec![vec![1.0]], vec![vec![-0.0]])?;
        assert_ne!(first.identity(), changed.identity());
        assert!(FfnCompensation::new(vec![], vec![]).is_err());
        assert!(FfnCompensation::new(vec![vec![1.0]], vec![vec![f32::NAN]]).is_err());
        assert!(FfnCompensation::new(vec![vec![1.0]; 17], vec![vec![1.0]; 17]).is_err());
        assert!(FfnCompensation::new(vec![vec![1.0], vec![1.0, 2.0]], vec![vec![1.0]; 2]).is_err());
        Ok(())
    }
}
