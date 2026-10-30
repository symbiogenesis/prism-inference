//! Explicit rank-one residual-writer and embedding corrections.
//!
//! These operations preserve the supplied adapter's original activation basis.
//! They do not establish that its authoring basis matches a particular model,
//! and do not enable adapters in the private Swift HTTP service.

use std::{collections::BTreeMap, path::Path};

use bonsai_gguf::MetadataValue;

use super::{Model, Tensor};
use crate::{
    Result, invalid,
    kernels::{MatvecBackend, MatvecScratch, matvec_into, row_into},
};

/// Exact approved Orca adapter file length.
pub const SWIFT_ADAPTER_BYTES: u64 = 9_682_464;
/// Exact approved Orca adapter file SHA-256.
pub const SWIFT_ADAPTER_SHA256: &str =
    "f1669534803d340a496015f5c45125f3437b4d13ec764f40e34488ce83967f42";
/// Conservative retained header and tensor budget for the approved adapter.
pub const SWIFT_ADAPTER_MEMORY_BYTES: usize = 16 * 1024 * 1024;
const MAX_PAIRS: usize = 513;

struct Factors {
    a: Tensor,
    b: Tensor,
}

/// An immutable, explicitly admitted rank-one adapter for one base model.
///
/// Only embedding and residual-writing attention/SSM/FFN factors are supported.
/// The numerical implementation follows the two matrix products, scale and add
/// stages separately. Structural admission is not full-model qualification.
pub struct RankOneAdapter {
    identity: String,
    base_identity: String,
    alpha: f32,
    factors: BTreeMap<String, Factors>,
}

/// Reusable bounded correction storage, independent of base projection scratch.
#[derive(Default)]
pub(crate) struct Scratch {
    rank: Vec<f32>,
    correction: Vec<f32>,
    matvec: MatvecScratch,
}

impl Scratch {
    pub(crate) fn new(backend: MatvecBackend, pinned: bool) -> Self {
        Self {
            rank: Vec::new(),
            correction: Vec::new(),
            matvec: if pinned {
                MatvecScratch::with_pinned_backend(backend)
            } else {
                MatvecScratch::with_backend(backend)
            },
        }
    }
}

impl RankOneAdapter {
    /// Loads the exact approved artifact for the exact original Swift PQ2 model.
    ///
    /// File verification precedes payload publication. Identical immutable B
    /// tensors share storage only after kind, layout and bytes match. This does
    /// not qualify grouped SSM authoring basis, outputs, quality or resources.
    ///
    /// # Errors
    /// Rejects identity/metadata/shape drift, nonfinite payloads, missing/extra
    /// factors, allocation failures and non-PQ2 base identity.
    pub fn load_swift(path: &Path, base: &Model) -> Result<Self> {
        Self::load_swift_representation(path, base, super::swift::Representation::Pq2)
    }

    /// Loads the approved adapter for one explicitly selected current Swift artifact.
    ///
    /// PQ2 and its exact current-weight PTQ1 export use the same factor inventory
    /// and original activation basis. This checks the selected base identity;
    /// callers must obtain its executable weights through the strict Swift loader.
    /// The older published PTQ1 checkpoint is never substituted for this export.
    ///
    /// # Errors
    /// Rejects a base identity outside the selected representation and the same
    /// artifact, inventory, shape and resource errors as [`Self::load_swift`].
    pub fn load_swift_representation(
        path: &Path,
        base: &Model,
        representation: super::swift::Representation,
    ) -> Result<Self> {
        if base.identity != representation.model_sha256() {
            return Err(invalid(
                "approved Swift adapter requires the exact selected current-weight base",
            ));
        }
        let mut model = Model::load(
            path,
            SWIFT_ADAPTER_BYTES,
            SWIFT_ADAPTER_SHA256,
            SWIFT_ADAPTER_MEMORY_BYTES,
        )?;
        model.share_lora_b();
        let adapter = Self::from_model(model, base)?;
        if adapter.factors.len() != 129 || adapter.alpha.to_bits() != 1.0_f32.to_bits() {
            return Err(invalid(
                "approved Swift adapter inventory or alpha mismatch",
            ));
        }
        Ok(adapter)
    }

    /// Validates already loaded factors against their exact base tensor shapes.
    ///
    /// Callers own file authenticity. The returned identity binds the supplied
    /// model identities; it does not attest that a manually constructed `Model`
    /// was verified. No permutation or transform is applied to `LoRA` A/B.
    ///
    /// # Errors
    /// Rejects other architectures/types, non-F32 or non-rank-one factors,
    /// unsupported/absent targets, incomplete pairs, nonfinite values, invalid
    /// alpha, more than 513 pairs, or empty identities.
    pub fn from_model(model: Model, base: &Model) -> Result<Self> {
        for (metadata, key, expected) in [
            (&model.metadata, "general.architecture", "qwen35"),
            (&base.metadata, "general.architecture", "qwen35"),
            (&model.metadata, "general.type", "adapter"),
            (&model.metadata, "adapter.type", "lora"),
        ] {
            if metadata.get(key).and_then(MetadataValue::as_str) != Some(expected) {
                return Err(invalid("unsupported rank-one adapter metadata"));
            }
        }
        let Some(MetadataValue::F32(alpha)) = model.metadata.get("adapter.lora_alpha") else {
            return Err(invalid("rank-one adapter requires an F32 alpha"));
        };
        if !alpha.is_finite()
            || *alpha <= 0.0
            || model.identity.is_empty()
            || base.identity.is_empty()
            || model.tensors.is_empty()
            || model.tensors.len() > MAX_PAIRS * 2
            || !model.tensors.len().is_multiple_of(2)
        {
            return Err(invalid(
                "invalid rank-one adapter alpha, identity or inventory",
            ));
        }
        let mut factors = BTreeMap::new();
        for (name, a) in &model.tensors {
            if let Some(target) = name.strip_suffix(".lora_a") {
                let base_tensor = base
                    .tensors
                    .get(target)
                    .filter(|_| supported_target(target))
                    .ok_or_else(|| invalid("unsupported rank-one adapter target"))?;
                let b = model
                    .tensors
                    .get(&format!("{target}.lora_b"))
                    .ok_or_else(|| invalid("rank-one adapter B factor missing"))?;
                let [input, output] = base_tensor.dimensions.as_slice() else {
                    return Err(invalid("rank-one adapter target must be a matrix"));
                };
                let a_shape = if target == "token_embd.weight" {
                    [1, *output]
                } else {
                    [*input, 1]
                };
                validate_factor(a, name, &a_shape)?;
                validate_factor(
                    b,
                    &format!("{target}.lora_b"),
                    &[
                        1,
                        if target == "token_embd.weight" {
                            *input
                        } else {
                            *output
                        },
                    ],
                )?;
                factors.insert(
                    target.to_owned(),
                    Factors {
                        a: a.clone(),
                        b: b.clone(),
                    },
                );
            } else if let Some(target) = name.strip_suffix(".lora_b") {
                if !model.tensors.contains_key(&format!("{target}.lora_a")) {
                    return Err(invalid("rank-one adapter A factor missing"));
                }
            } else {
                return Err(invalid("unsupported rank-one adapter tensor name"));
            }
        }
        if factors.len() * 2 != model.tensors.len() {
            return Err(invalid("incomplete rank-one adapter inventory"));
        }
        Ok(Self {
            identity: model.identity,
            base_identity: base.identity.clone(),
            alpha: *alpha,
            factors,
        })
    }

    /// Identity of the supplied adapter model or its verified GGUF file.
    #[must_use]
    pub fn identity(&self) -> &str {
        &self.identity
    }

    /// Number of complete admitted factor pairs.
    #[must_use]
    pub fn target_count(&self) -> usize {
        self.factors.len()
    }

    pub(crate) fn base_identity(&self) -> &str {
        &self.base_identity
    }

    pub(crate) fn targets(&self) -> impl Iterator<Item = &str> {
        self.factors.keys().map(String::as_str)
    }

    pub(crate) fn validate_scale(scale: f32) -> Result<()> {
        if scale.is_finite() && (0.0..=2.0).contains(&scale) {
            Ok(())
        } else {
            Err(invalid("rank-one adapter scale must be finite and in 0..2"))
        }
    }

    pub(crate) fn linear(
        &self,
        target: &str,
        input: &[f32],
        scale: f32,
        output: &mut [f32],
        scratch: &mut Scratch,
    ) -> Result<()> {
        let Some(factors) = self.factors.get(target).filter(|_| scale != 0.0) else {
            return Ok(());
        };
        matvec_into(&factors.a, input, &mut scratch.rank, &mut scratch.matvec)?;
        matvec_into(
            &factors.b,
            &scratch.rank,
            &mut scratch.correction,
            &mut scratch.matvec,
        )?;
        self.finish(scale, output, &mut scratch.correction)
    }

    pub(crate) fn embedding(
        &self,
        token: usize,
        scale: f32,
        output: &mut [f32],
        scratch: &mut Scratch,
    ) -> Result<()> {
        let Some(factors) = self
            .factors
            .get("token_embd.weight")
            .filter(|_| scale != 0.0)
        else {
            return Ok(());
        };
        row_into(&factors.a, token, &mut scratch.rank)?;
        matvec_into(
            &factors.b,
            &scratch.rank,
            &mut scratch.correction,
            &mut scratch.matvec,
        )?;
        self.finish(scale, output, &mut scratch.correction)
    }

    #[allow(
        clippy::suboptimal_flops,
        reason = "Preserve the native product, scale and add as separately rounded F32 stages."
    )]
    fn finish(&self, scale: f32, output: &mut [f32], correction: &mut [f32]) -> Result<()> {
        if output.len() != correction.len() {
            return Err(invalid("rank-one correction extent mismatch"));
        }
        let scale = scale * self.alpha;
        for (&base, correction) in output.iter().zip(correction.iter_mut()) {
            *correction *= scale;
            *correction += base;
            if !correction.is_finite() {
                return Err(invalid("nonfinite rank-one correction"));
            }
        }
        output.copy_from_slice(correction);
        Ok(())
    }
}

fn supported_target(name: &str) -> bool {
    if name == "token_embd.weight" {
        return true;
    }
    let Some((layer, suffix)) = name
        .strip_prefix("blk.")
        .and_then(|name| name.split_once('.'))
    else {
        return false;
    };
    !layer.is_empty()
        && layer.bytes().all(|byte| byte.is_ascii_digit())
        && matches!(
            suffix,
            "attn_output.weight" | "ssm_out.weight" | "ffn_down.weight"
        )
}

fn validate_factor(tensor: &Tensor, name: &str, shape: &[usize]) -> Result<()> {
    let bytes = shape
        .iter()
        .try_fold(4_usize, |bytes, dimension| bytes.checked_mul(*dimension));
    if tensor.name != name
        || tensor.kind != 0
        || tensor.dimensions != shape
        || shape.contains(&0)
        || bytes != Some(tensor.data.len())
        || tensor
            .data
            .as_chunks::<4>()
            .0
            .iter()
            .any(|bytes| !f32::from_le_bytes(*bytes).is_finite())
    {
        return Err(invalid("invalid rank-one adapter factor"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{collections::BTreeMap, sync::Arc};

    use bonsai_gguf::MetadataValue;

    use super::{Model, RankOneAdapter, Result, Scratch, Tensor};
    use crate::kernels::MatvecBackend;
    use crate::model::swift::Representation;

    fn tensor(name: &str, values: &[f32]) -> Tensor {
        Tensor {
            name: name.into(),
            dimensions: vec![1, values.len()],
            kind: 0,
            data: Arc::from(
                values
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect::<Vec<_>>(),
            ),
        }
    }

    fn adapter(a: f32, b: f32) -> Result<RankOneAdapter> {
        let target = "blk.0.ffn_down.weight";
        let base = Model {
            identity: "base".into(),
            metadata: BTreeMap::from([(
                "general.architecture".into(),
                MetadataValue::from("qwen35"),
            )]),
            tensors: BTreeMap::from([(target.into(), tensor(target, &[0.0]))]),
        };
        let factors = Model {
            identity: "adapter".into(),
            metadata: BTreeMap::from([
                ("general.architecture".into(), MetadataValue::from("qwen35")),
                ("general.type".into(), MetadataValue::from("adapter")),
                ("adapter.type".into(), MetadataValue::from("lora")),
                ("adapter.lora_alpha".into(), MetadataValue::from(1.0_f32)),
            ]),
            tensors: BTreeMap::from([
                (
                    format!("{target}.lora_a"),
                    tensor(&format!("{target}.lora_a"), &[a]),
                ),
                (
                    format!("{target}.lora_b"),
                    tensor(&format!("{target}.lora_b"), &[b]),
                ),
            ]),
        };
        RankOneAdapter::from_model(factors, &base)
    }

    #[test]
    fn rank_one_selected_current_weight_base_keeps_default_and_adapter_verification_strict()
    -> Result<()> {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "PrismAdapterIdentity-{}-{nonce}.gguf",
            std::process::id()
        ));
        std::fs::write(&path, b"unapproved adapter bytes")?;
        let mut base = Model {
            identity: String::new(),
            metadata: BTreeMap::new(),
            tensors: BTreeMap::new(),
        };
        for representation in [Representation::Pq2, Representation::CurrentRepackedPtq1] {
            base.identity = representation.model_sha256().into();
            let error = RankOneAdapter::load_swift_representation(&path, &base, representation)
                .err()
                .ok_or("unapproved adapter was admitted")?;
            assert!(error.to_string().contains("artifact size mismatch"));
            let other = if representation == Representation::Pq2 {
                Representation::CurrentRepackedPtq1
            } else {
                Representation::Pq2
            };
            let error = RankOneAdapter::load_swift_representation(&path, &base, other)
                .err()
                .ok_or("mismatched base representation was admitted")?;
            assert!(error.to_string().contains("selected current-weight base"));
        }
        assert!(RankOneAdapter::load_swift(&path, &base).is_err());
        base.identity = "older-published-PTQ1-checkpoint".into();
        assert!(
            RankOneAdapter::load_swift_representation(
                &path,
                &base,
                Representation::CurrentRepackedPtq1
            )
            .is_err()
        );
        std::fs::remove_file(path)?;
        Ok(())
    }

    #[test]
    fn rank_one_products_round_before_scale_and_base_addition() -> Result<()> {
        let adapter = adapter(4097.0, 1.0)?;
        let mut scratch = Scratch::new(MatvecBackend::Scalar, true);
        let mut output = [-16_785_408.0];
        adapter.linear(
            "blk.0.ffn_down.weight",
            &[4097.0],
            1.0,
            &mut output,
            &mut scratch,
        )?;
        // The unrounded product is 16_785_409; fusion would incorrectly give 1.
        assert_eq!(output, [0.0]);
        assert_eq!(scratch.rank.len(), 1);
        assert_eq!(scratch.correction.len(), 1);
        Ok(())
    }

    #[test]
    fn rank_one_zero_skips_nonfinite_branch_and_failures_preserve_output() -> Result<()> {
        let adapter = adapter(f32::MAX, 1.0)?;
        let mut scratch = Scratch::new(MatvecBackend::Scalar, false);
        for scale in [0.0, -0.0] {
            let mut output = [-0.0];
            adapter.linear(
                "blk.0.ffn_down.weight",
                &[f32::MAX],
                scale,
                &mut output,
                &mut scratch,
            )?;
            assert_eq!(output[0].to_bits(), (-0.0_f32).to_bits());
            assert_eq!(scratch.rank, Vec::<f32>::new());
            assert_eq!(scratch.correction, Vec::<f32>::new());
        }
        let mut output = [3.0];
        assert!(
            adapter
                .linear(
                    "blk.0.ffn_down.weight",
                    &[1.0],
                    2.0,
                    &mut output,
                    &mut scratch
                )
                .is_err()
        );
        assert_eq!(output, [3.0]);
        Ok(())
    }
}
