//! Scoped projection execution with unchanged graph-state transactions.

use super::{Attention, Qwen35, Qwen35Checkpoint};
use crate::{Decoder, Result, invalid, kernels::Pq2ProjectionExecutor, model::Tensor};
use bonsai_native_hash::Hasher;

/// Checkpoint handle for [`ProjectedQwen35`], compatible with [`Qwen35Checkpoint`]
/// while binding the active deterministic projection-plan digest when captured
/// inside a projection scope.
pub type ProjectedCheckpoint = Qwen35Checkpoint;

fn hash_u64(hasher: &mut Hasher, value: u64) -> Result<()> {
    hasher.update(&value.to_le_bytes())?;
    Ok(())
}

fn hash_usize(hasher: &mut Hasher, value: usize) -> Result<()> {
    hash_u64(hasher, u64::try_from(value)?)
}

fn hash_tensor_plan(
    hasher: &mut Hasher,
    decoder: &Qwen35,
    executor: &dyn Pq2ProjectionExecutor,
    tensor: &Tensor,
) -> Result<()> {
    hash_usize(hasher, tensor.name.len())?;
    hasher.update(tensor.name.as_bytes())?;
    hasher.update(&tensor.kind.to_le_bytes())?;
    hash_usize(hasher, tensor.dimensions.len())?;
    for &dimension in &tensor.dimensions {
        hash_usize(hasher, dimension)?;
    }
    let (rotated, grouped) = decoder.transforms.basis(tensor);
    hasher.update(&[
        u8::from(rotated),
        u8::from(grouped),
        u8::from(executor.supports(tensor)),
        u8::from(executor.supports_f32(tensor)),
        u8::from(executor.supports_q8(tensor)),
    ])?;
    Ok(())
}

fn projection_plan_digest(
    decoder: &Qwen35,
    executor: &dyn Pq2ProjectionExecutor,
    work: Option<Work>,
) -> Result<[u8; 32]> {
    let mut hasher = Hasher::new()?;
    hasher.update(b"bonsai.prism.projected_plan.v1")?;
    hash_usize(&mut hasher, decoder.identity.len())?;
    hasher.update(decoder.identity.as_bytes())?;
    let placement = executor.identity();
    hash_usize(&mut hasher, placement.len())?;
    hasher.update(placement.as_bytes())?;
    hasher.update(&[
        u8::from(work.is_some()),
        u8::from(!decoder.adapters().is_any_loaded()),
    ])?;
    hash_u64(&mut hasher, work.map_or(0, |plan| plan.hidden))?;
    hash_u64(&mut hasher, work.map_or(0, |plan| plan.output))?;
    hash_usize(&mut hasher, decoder.layers.len())?;
    for (layer_index, layer) in decoder.layers.iter().enumerate() {
        hash_usize(&mut hasher, layer_index)?;
        match &layer.attention {
            Attention::Full(attention) => {
                hasher.update(&[0])?;
                for tensor in [
                    &attention.query,
                    &attention.key,
                    &attention.value,
                    &attention.output,
                ] {
                    hash_tensor_plan(&mut hasher, decoder, executor, tensor)?;
                }
            }
            Attention::Recurrent(attention) => {
                hasher.update(&[1])?;
                for tensor in [
                    &attention.qkv,
                    &attention.gate,
                    &attention.beta,
                    &attention.alpha,
                    &attention.output,
                ] {
                    hash_tensor_plan(&mut hasher, decoder, executor, tensor)?;
                }
            }
        }
        for tensor in [&layer.up, &layer.gate, &layer.down] {
            hash_tensor_plan(&mut hasher, decoder, executor, tensor)?;
        }
        let hadamard = decoder.transforms.ffn_chain_hadamard_block_for(&layer.down);
        let (hadamard_tag, hadamard_block) = match hadamard {
            None => (0_u8, 0_u32),
            Some(None) => (1_u8, 0_u32),
            Some(Some(block)) => (2_u8, block),
        };
        let chain_tensors = [&layer.gate, &layer.up, &layer.down];
        let adapters = decoder.adapters();
        let can_chain = !adapters.affects_any(&chain_tensors);
        let can_swiglu = !adapters.affects_any(&[&layer.gate, &layer.up]);
        hasher.update(&[
            hadamard_tag,
            u8::from(can_chain && executor.supports_f32_ffn_chain(chain_tensors)),
            u8::from(can_swiglu && executor.supports_f32_ffn_swiglu(&layer.gate, &layer.up)),
        ])?;
        hasher.update(&hadamard_block.to_le_bytes())?;
    }
    hash_tensor_plan(&mut hasher, decoder, executor, &decoder.output)?;
    Ok(hasher.finish()?)
}

#[derive(Clone, Copy, Default)]
struct Work {
    hidden: u64,
    output: u64,
}

fn projection_operations(executor: &dyn Pq2ProjectionExecutor, tensor: &Tensor) -> Result<u64> {
    let columns = tensor.dimensions[0];
    let selected = (matches!(tensor.kind, 142 | 143)
        && columns.is_multiple_of(128)
        && executor.supports_f32(tensor))
        || (tensor.kind == 142 && columns.is_multiple_of(256) && executor.supports(tensor))
        || (matches!(tensor.kind, 142 | 143) && executor.supports_q8(tensor));
    if !selected {
        return Ok(0);
    }
    executor
        .operation_count(tensor)
        .filter(|&count| count > 0)
        .ok_or_else(|| invalid("bounded projection executor omitted selected work count"))
}

impl Work {
    fn new(decoder: &Qwen35, executor: &dyn Pq2ProjectionExecutor) -> Result<Option<Self>> {
        if executor.operation_capacity().is_none() {
            return Ok(None);
        }
        let mut work = Self::default();
        let mut extra_hidden = 0_u64;
        let mut add = |tensor: &Tensor| -> Result<()> {
            work.hidden = work
                .hidden
                .checked_add(projection_operations(executor, tensor)?)
                .ok_or_else(|| invalid("projection work count overflow"))?;
            Ok(())
        };
        for layer in &decoder.layers {
            match &layer.attention {
                Attention::Full(attention) => {
                    for tensor in [
                        &attention.query,
                        &attention.key,
                        &attention.value,
                        &attention.output,
                    ] {
                        add(tensor)?;
                    }
                }
                Attention::Recurrent(attention) => {
                    for tensor in [
                        &attention.qkv,
                        &attention.gate,
                        &attention.beta,
                        &attention.alpha,
                        &attention.output,
                    ] {
                        add(tensor)?;
                    }
                }
            }
            for tensor in [&layer.up, &layer.gate, &layer.down] {
                add(tensor)?;
            }
            if decoder.transforms.basis(&layer.gate) == decoder.transforms.basis(&layer.up) {
                let chain_tensors = [&layer.gate, &layer.up, &layer.down];
                let adapters = decoder.adapters();
                let can_chain = !adapters.affects_any(&chain_tensors);
                let can_swiglu = !adapters.affects_any(&[&layer.gate, &layer.up]);
                if let Some(hadamard_block) =
                    decoder.transforms.ffn_chain_hadamard_block_for(&layer.down)
                    && can_chain
                    && executor.supports_f32_ffn_chain(chain_tensors)
                {
                    let extra = 2_u64 + u64::from(hadamard_block.is_some());
                    extra_hidden = extra_hidden
                        .checked_add(extra)
                        .ok_or_else(|| invalid("projection work count overflow"))?;
                } else if can_swiglu && executor.supports_f32_ffn_swiglu(&layer.gate, &layer.up) {
                    extra_hidden = extra_hidden
                        .checked_add(1)
                        .ok_or_else(|| invalid("projection work count overflow"))?;
                }
            }
        }
        work.hidden = work
            .hidden
            .checked_add(extra_hidden)
            .ok_or_else(|| invalid("projection work count overflow"))?;
        work.output = projection_operations(executor, &decoder.output)?;
        Ok(Some(work))
    }

    fn operations(self, ingest: usize, evaluate: usize) -> Result<u64> {
        let ingest = u64::try_from(ingest)?;
        let evaluate = u64::try_from(evaluate)?;
        ingest
            .checked_add(evaluate)
            .and_then(|tokens| tokens.checked_mul(self.hidden))
            .and_then(|hidden| {
                evaluate
                    .checked_mul(self.output)
                    .and_then(|output| hidden.checked_add(output))
            })
            .ok_or_else(|| invalid("projection request work count overflow"))
    }
}

/// Experimental decoder borrowing one synchronous projection executor.
///
/// Construct this inside the owned native session callback, then place it in an
/// ordinary [`crate::runtime::Runtime`] for complete request checkpoints. The
/// borrow keeps one session alive across all eligible projections, tokens and
/// request phases without per-projection module open/close. It neither extends a
/// terminal native session's operation/deadline limits nor enables a service
/// profile. Unsupported projection shapes stay on CPU before submission.
///
/// A work-budget refusal occurs before evaluation and leaves this scope healthy.
/// Any error after admitted evaluation starts permanently closes this scope.
/// Reset or restoration cannot authorize further native work; the owner must finish verified teardown
/// or end its supervised process. Device consumers and resources are outside
/// graph checkpoints and must never be restored by replaying a failed projection.
/// Checkpoints intentionally contain the same CPU-owned committed state as the
/// underlying graph. A checkpoint taken from that graph before wrapping remains
/// valid while this scope is healthy; foreign graph/model provenance still
/// rejects. Native queues, in-flight buffers and device resources are excluded.
pub struct ProjectedQwen35<'a> {
    decoder: Qwen35,
    executor: &'a mut dyn Pq2ProjectionExecutor,
    identity: String,
    faulted: bool,
    work: Option<Work>,
    plan_digest: [u8; 32],
}

impl Qwen35 {
    /// Borrow one explicit executor while retaining native graph and adapter order.
    ///
    /// This API selects no device, loads no driver and admits no production
    /// accelerator. The caller independently owns signed deployment, exact device
    /// identity, resource admission, an absolute process deadline and teardown.
    /// The executor identity distinguishes scoped runtime caches from CPU placement.
    ///
    /// # Errors
    /// Rejects invalid identities, allocation or missing bounded work counts.
    pub fn with_projection_executor(
        self,
        executor: &mut dyn Pq2ProjectionExecutor,
    ) -> Result<ProjectedQwen35<'_>> {
        let placement = executor.identity();
        if placement.is_empty()
            || placement.len() > 256
            || !placement.bytes().all(|byte| byte.is_ascii_graphic())
        {
            return Err(invalid("invalid projection executor identity"));
        }
        let work = Work::new(&self, executor)?;
        let plan_digest = projection_plan_digest(&self, executor, work)?;
        let mut identity = String::new();
        identity.try_reserve_exact(self.identity.len() + placement.len() + 32)?;
        identity.push_str(&self.identity);
        identity.push_str(":external-pq2-v1:");
        identity.push_str(placement);
        Ok(ProjectedQwen35 {
            decoder: self,
            executor,
            identity,
            faulted: false,
            work,
            plan_digest,
        })
    }
}

impl ProjectedQwen35<'_> {
    /// Read-only underlying graph for state and CPU-fallback diagnostics.
    #[must_use]
    pub const fn decoder(&self) -> &Qwen35 {
        &self.decoder
    }

    /// Whether this scope rejected work and requires terminal owner teardown.
    #[must_use]
    pub const fn is_faulted(&self) -> bool {
        self.faulted
    }

    fn available(&self) -> Result<()> {
        if self.faulted {
            Err(invalid(
                "projection executor scope requires terminal teardown",
            ))
        } else {
            Ok(())
        }
    }

    fn evaluate_token(&mut self, token: u32, logits: Option<&mut Vec<f32>>) -> Result<()> {
        self.available()?;
        // Refuse a complete token before any projection or state delta. This
        // refusal leaves the scope healthy; submitted failures remain terminal.
        self.admit_work(usize::from(logits.is_none()), usize::from(logits.is_some()))?;
        let result =
            self.decoder
                .evaluate_token_with_executor(token, logits, Some(self.executor), None);
        self.faulted = result.is_err();
        result
    }
}

impl Decoder for ProjectedQwen35<'_> {
    type Checkpoint = ProjectedCheckpoint;

    fn identity(&self) -> &str {
        &self.identity
    }
    fn vocab_size(&self) -> usize {
        self.decoder.vocab_size()
    }
    fn checkpoint_bytes(&self) -> usize {
        self.decoder.checkpoint_bytes()
    }
    fn admit_work(&self, ingest_tokens: usize, evaluate_tokens: usize) -> Result<()> {
        self.available()?;
        match (self.work, self.executor.operation_capacity()) {
            (Some(work), Some(remaining)) => {
                if work.operations(ingest_tokens, evaluate_tokens)? > remaining {
                    return Err(invalid(
                        "projection request exceeds remaining native operation budget",
                    ));
                }
            }
            (None, None) => {}
            _ => {
                return Err(invalid(
                    "projection executor changed its operation budget contract",
                ));
            }
        }
        Ok(())
    }
    fn checkpoint(&self) -> Result<Self::Checkpoint> {
        self.available()?;
        let mut checkpoint = self.decoder.checkpoint()?;
        checkpoint.projection_plan_digest = Some(self.plan_digest);
        Ok(checkpoint)
    }
    fn checkpoint_into(&self, target: &mut Self::Checkpoint) -> Result<()> {
        self.available()?;
        self.decoder.checkpoint_into(target)?;
        target.projection_plan_digest = Some(self.plan_digest);
        Ok(())
    }
    fn restore(&mut self, checkpoint: &Self::Checkpoint) -> Result<()> {
        self.available()?;
        if let Some(saved_digest) = checkpoint.projection_plan_digest
            && saved_digest != self.plan_digest
        {
            return Err(invalid("checkpoint belongs to a different projection plan"));
        }
        self.decoder.restore(checkpoint)
    }
    fn reset(&mut self) -> Result<()> {
        self.available()?;
        self.decoder.reset()
    }
    fn evaluate(&mut self, token: u32) -> Result<Vec<f32>> {
        let mut logits = Vec::new();
        self.evaluate_into(token, &mut logits)?;
        Ok(logits)
    }
    fn evaluate_into(&mut self, token: u32, output: &mut Vec<f32>) -> Result<()> {
        let result = self.evaluate_token(token, Some(output));
        if result.is_err() {
            output.clear();
        }
        result
    }
    fn evaluate_finite_into<'a>(
        &mut self,
        token: u32,
        output: &'a mut Vec<f32>,
    ) -> Result<crate::sampling::FiniteLogits<'a>> {
        self.evaluate_into(token, output)?;
        // The scoped executor uses the same complete-output validation and
        // atomic graph commit as Qwen35; device loss remains latched closed.
        crate::sampling::FiniteLogits::from_validated(output)
    }
    fn ingest(&mut self, token: u32) -> Result<()> {
        self.evaluate_token(token, None)
    }
    fn ingest_block(&mut self, tokens: &[u32], check: &dyn Fn() -> Result<()>) -> Result<()> {
        self.available()?;
        self.admit_work(tokens.len(), 0)?;
        let result =
            self.decoder
                .ingest_prefill_block_with_executor(tokens, check, Some(self.executor));
        self.faulted = result.is_err();
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    struct PlanTestExecutor {
        id: &'static str,
        supports_f32: bool,
        supports_ffn_fusion: bool,
        bounded: bool,
    }

    impl Pq2ProjectionExecutor for PlanTestExecutor {
        fn identity(&self) -> &str {
            self.id
        }

        fn operation_capacity(&self) -> Option<u64> {
            self.bounded.then_some(1_000)
        }

        fn operation_count(&self, _tensor: &Tensor) -> Option<u64> {
            self.bounded.then_some(1)
        }

        fn supports(&self, _tensor: &Tensor) -> bool {
            true
        }

        fn supports_f32(&self, _tensor: &Tensor) -> bool {
            self.supports_f32
        }

        fn supports_f32_ffn_swiglu(&self, _gate: &Tensor, _up: &Tensor) -> bool {
            self.supports_ffn_fusion
        }

        fn supports_f32_ffn_chain(&self, _tensors: [&Tensor; 3]) -> bool {
            self.supports_ffn_fusion
        }

        fn project(
            &mut self,
            _tensor: &Tensor,
            _activation: &[bonsai_ternary::Q8KBlock],
            _fused: bool,
            output: &mut Vec<f32>,
        ) -> Result<()> {
            output.clear();
            Ok(())
        }
    }

    fn stub_tensor(name: &str) -> Tensor {
        Tensor {
            name: name.to_owned(),
            dimensions: vec![128, 128],
            kind: 142,
            data: Arc::<[u8]>::from([]).into(),
        }
    }

    fn tiny_decoder(provenance: Arc<()>) -> Result<Qwen35> {
        Ok(Qwen35 {
            provenance,
            identity: "projection-plan-digest-fixture".to_owned(),
            geometry: super::super::Geometry {
                embedding: 128,
                feed_forward: 128,
                heads: 1,
                kv_heads: 1,
                head: 128,
                rope: 128,
                rope_sections: [64, 0, 0, 0],
                rope_base: 10_000.0,
                epsilon: 1e-6,
                ssm_head: 128,
                ssm_keys: 1,
                ssm_values: 1,
                convolution: 4,
            },
            arithmetic: super::super::Arithmetic::Source,
            embedding: stub_tensor("token_embd.weight"),
            output: stub_tensor("output.weight"),
            output_norm: vec![1.0; 128],
            transforms: super::super::Transforms::default(),
            adapter: None,
            adapter_scale: 0.0,
            multi_rank_adapter: None,
            multi_rank_adapter_scale: 0.0,
            layers: vec![super::super::Layer {
                norm: vec![1.0; 128],
                post_norm: vec![1.0; 128],
                attention: Attention::Full(super::super::FullAttention {
                    query: stub_tensor("blk.0.attn_q.weight"),
                    key: stub_tensor("blk.0.attn_k.weight"),
                    value: stub_tensor("blk.0.attn_v.weight"),
                    output: stub_tensor("blk.0.attn_output.weight"),
                    query_norm: vec![1.0; 128],
                    key_norm: vec![1.0; 128],
                }),
                up: stub_tensor("blk.0.ffn_up.weight"),
                gate: stub_tensor("blk.0.ffn_gate.weight"),
                down: stub_tensor("blk.0.ffn_down.weight"),
            }],
            state: vec![super::super::LayerState::Full {
                keys: super::super::KvCache::new(128, 8)?,
                values: super::super::KvCache::new(128, 8)?,
            }],
            workspace: super::super::Workspace::default(),
            context_limit: 8,
            position: 0,
            mrope_position: 0,
            state_consistent: true,
            vocabulary: 2,
            kv_cache_policy: super::super::KvCachePolicy::F16,
            narrow_last_layer: false,
            tiled_prefill: None,
            projection_executor: None,
        })
    }

    #[test]
    fn projection_plan_digest_rejects_mismatched_plan_and_accepts_prewrap_cpu_checkpoint()
    -> Result<()> {
        let provenance = Arc::new(());
        let decoder_a = tiny_decoder(Arc::clone(&provenance))?;
        let cpu_checkpoint = decoder_a.checkpoint()?;
        let mut reused = decoder_a.checkpoint()?;
        assert!(cpu_checkpoint.projection_plan_digest.is_none());

        let mut exec_a = PlanTestExecutor {
            id: "plan-a",
            supports_f32: false,
            supports_ffn_fusion: false,
            bounded: false,
        };
        let proj_checkpoint = {
            let mut projected = decoder_a.with_projection_executor(&mut exec_a)?;
            let first = projected.checkpoint()?;
            projected.checkpoint_into(&mut reused)?;
            assert_eq!(first.projection_plan_digest, reused.projection_plan_digest);
            assert_eq!(first.projection_plan_digest, Some(projected.plan_digest));
            projected.restore(&cpu_checkpoint)?;
            first
        };

        let mut exec_b = PlanTestExecutor {
            id: "plan-b",
            supports_f32: false,
            supports_ffn_fusion: false,
            bounded: false,
        };
        let decoder_b = tiny_decoder(Arc::clone(&provenance))?;
        let mut projected_b = decoder_b.with_projection_executor(&mut exec_b)?;
        assert_ne!(
            Some(projected_b.plan_digest),
            proj_checkpoint.projection_plan_digest
        );
        assert!(projected_b.restore(&proj_checkpoint).is_err());

        let mut exec_c = PlanTestExecutor {
            id: "plan-a",
            supports_f32: true,
            supports_ffn_fusion: false,
            bounded: false,
        };
        let decoder_c = tiny_decoder(provenance)?;
        let mut projected_c = decoder_c.with_projection_executor(&mut exec_c)?;
        assert_ne!(
            Some(projected_c.plan_digest),
            proj_checkpoint.projection_plan_digest
        );
        assert!(projected_c.restore(&proj_checkpoint).is_err());
        Ok(())
    }

    fn adapter_targeting(target: &Tensor) -> Result<crate::model::lora::RankOneAdapter> {
        use std::collections::BTreeMap;
        let base = crate::model::Model {
            identity: "projection-adapter-base".to_owned(),
            metadata: BTreeMap::from([(
                "general.architecture".to_owned(),
                bonsai_gguf::MetadataValue::from("qwen35"),
            )]),
            tensors: BTreeMap::from([(target.name.clone(), target.clone())]),
        };
        let factor = |name: &str| Tensor {
            name: name.to_owned(),
            dimensions: vec![128, 1],
            kind: 0,
            data: Arc::<[u8]>::from(1.0_f32.to_le_bytes().repeat(128)).into(),
        };
        let factor_b = |name: &str| Tensor {
            name: name.to_owned(),
            dimensions: vec![1, 128],
            kind: 0,
            data: Arc::<[u8]>::from(1.0_f32.to_le_bytes().repeat(128)).into(),
        };
        let factors = crate::model::Model {
            identity: "projection-adapter-factors".to_owned(),
            metadata: BTreeMap::from([
                (
                    "general.architecture".to_owned(),
                    bonsai_gguf::MetadataValue::from("qwen35"),
                ),
                (
                    "general.type".to_owned(),
                    bonsai_gguf::MetadataValue::from("adapter"),
                ),
                (
                    "adapter.type".to_owned(),
                    bonsai_gguf::MetadataValue::from("lora"),
                ),
                (
                    "adapter.lora_alpha".to_owned(),
                    bonsai_gguf::MetadataValue::from(1.0_f32),
                ),
            ]),
            tensors: BTreeMap::from([
                (
                    format!("{}.lora_a", target.name),
                    factor(&format!("{}.lora_a", target.name)),
                ),
                (
                    format!("{}.lora_b", target.name),
                    factor_b(&format!("{}.lora_b", target.name)),
                ),
            ]),
        };
        crate::model::lora::RankOneAdapter::from_model(factors, &base)
    }

    #[test]
    fn projection_work_and_digest_track_per_layer_lora_ffn_fusion() -> Result<()> {
        let mut exec = PlanTestExecutor {
            id: "fused-ffn-budget",
            supports_f32: true,
            supports_ffn_fusion: true,
            bounded: true,
        };
        let provenance = Arc::new(());
        let mut dec_zero = tiny_decoder(Arc::clone(&provenance))?;
        dec_zero.adapter = Some(adapter_targeting(&dec_zero.layers[0].down)?);
        dec_zero.adapter_scale = 0.0;
        let zero_proj = dec_zero.with_projection_executor(&mut exec)?;
        assert_eq!(zero_proj.work.map(|w| w.hidden), Some(9));

        let mut dec_attn = tiny_decoder(Arc::clone(&provenance))?;
        if let Attention::Full(attn) = &dec_attn.layers[0].attention {
            dec_attn.adapter = Some(adapter_targeting(&attn.output)?);
        }
        dec_attn.adapter_scale = 1.3;
        let attn_proj = dec_attn.with_projection_executor(&mut exec)?;
        assert_eq!(attn_proj.work.map(|w| w.hidden), Some(9));
        let attn_digest = attn_proj.plan_digest;
        drop(attn_proj);

        let mut dec_down = tiny_decoder(provenance)?;
        dec_down.adapter = Some(adapter_targeting(&dec_down.layers[0].down)?);
        dec_down.adapter_scale = 1.3;
        let down_proj = dec_down.with_projection_executor(&mut exec)?;
        assert_eq!(down_proj.work.map(|w| w.hidden), Some(8));
        assert_ne!(attn_digest, down_proj.plan_digest);
        Ok(())
    }

    struct Ptq1PlanTestExecutor {
        supports_f32: bool,
        supports_ffn_fusion: bool,
    }

    impl Pq2ProjectionExecutor for Ptq1PlanTestExecutor {
        fn identity(&self) -> &'static str {
            "ptq1-plan-test"
        }

        fn operation_capacity(&self) -> Option<u64> {
            Some(100)
        }

        fn operation_count(&self, _tensor: &Tensor) -> Option<u64> {
            Some(2)
        }

        fn supports(&self, _tensor: &Tensor) -> bool {
            false
        }

        fn supports_f32(&self, _tensor: &Tensor) -> bool {
            self.supports_f32
        }

        fn supports_f32_ffn_swiglu(&self, _gate: &Tensor, _up: &Tensor) -> bool {
            self.supports_ffn_fusion
        }

        fn supports_f32_ffn_chain(&self, _tensors: [&Tensor; 3]) -> bool {
            self.supports_ffn_fusion
        }

        fn project(
            &mut self,
            _tensor: &Tensor,
            _activation: &[bonsai_ternary::Q8KBlock],
            _fused: bool,
            _output: &mut Vec<f32>,
        ) -> Result<()> {
            Err(invalid("unexpected Q8_K project"))
        }
    }

    #[test]
    fn projected_qwen35_accounts_work_and_digest_for_ptq1_supports_f32() -> Result<()> {
        let provenance = Arc::new(());
        let mut decoder = tiny_decoder(Arc::clone(&provenance))?;
        decoder.output.kind = 143;
        if let Attention::Full(attn) = &mut decoder.layers[0].attention {
            attn.query.kind = 143;
            attn.key.kind = 143;
            attn.value.kind = 143;
            attn.output.kind = 143;
        }
        decoder.layers[0].up.kind = 143;
        decoder.layers[0].gate.kind = 143;
        decoder.layers[0].down.kind = 143;

        let mut exec_on = Ptq1PlanTestExecutor {
            supports_f32: true,
            supports_ffn_fusion: true,
        };
        let proj_on = decoder.with_projection_executor(&mut exec_on)?;
        // 7 PTQ1 layer projections * 2 ops + 2 FFN chain extra hidden ops = 16; output = 2.
        assert_eq!(proj_on.work.map(|w| (w.hidden, w.output)), Some((16, 2)));
        assert!(proj_on.admit_work(2, 1).is_ok());
        let digest_on = proj_on.plan_digest;
        drop(proj_on);

        let mut decoder_off = tiny_decoder(provenance)?;
        decoder_off.output.kind = 143;
        if let Attention::Full(attn) = &mut decoder_off.layers[0].attention {
            attn.query.kind = 143;
            attn.key.kind = 143;
            attn.value.kind = 143;
            attn.output.kind = 143;
        }
        decoder_off.layers[0].up.kind = 143;
        decoder_off.layers[0].gate.kind = 143;
        decoder_off.layers[0].down.kind = 143;
        let mut exec_off = Ptq1PlanTestExecutor {
            supports_f32: false,
            supports_ffn_fusion: false,
        };
        let proj_off = decoder_off.with_projection_executor(&mut exec_off)?;
        assert_eq!(proj_off.work.map(|w| (w.hidden, w.output)), Some((0, 0)));
        assert_ne!(digest_on, proj_off.plan_digest);
        Ok(())
    }
}
