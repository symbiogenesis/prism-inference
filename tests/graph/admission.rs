//! Whole-request device work refusal precedes any new token or publication.

use super::*;
use bonsai_prism_runtime::kernels::Pq2ProjectionExecutor;
use bonsai_ternary::Q8KBlock;
use std::{cell::Cell, rc::Rc};

struct BoundedExecutor {
    remaining: Rc<Cell<u64>>,
    cost: Option<u64>,
    calls: Rc<Cell<usize>>,
}

impl Pq2ProjectionExecutor for BoundedExecutor {
    fn identity(&self) -> &'static str {
        "bounded-synthetic-projection-control"
    }
    fn supports(&self, _: &Tensor) -> bool {
        true
    }
    fn operation_capacity(&self) -> Option<u64> {
        Some(self.remaining.get())
    }
    fn operation_count(&self, _: &Tensor) -> Option<u64> {
        self.cost
    }
    fn project(
        &mut self,
        tensor: &Tensor,
        _: &[Q8KBlock],
        _: bool,
        output: &mut Vec<f32>,
    ) -> Result<()> {
        let cost = self
            .cost
            .ok_or_else(|| std::io::Error::other("missing work count"))?;
        self.remaining.set(
            self.remaining
                .get()
                .checked_sub(cost)
                .ok_or_else(|| std::io::Error::other("synthetic native budget exhaustion"))?,
        );
        self.calls.set(self.calls.get() + 1);
        output.clear();
        output.resize(tensor.dimensions[1], 0.0);
        Ok(())
    }
}

fn executor(remaining: u64, cost: Option<u64>) -> BoundedExecutor {
    BoundedExecutor {
        remaining: Rc::new(Cell::new(remaining)),
        cost,
        calls: Rc::new(Cell::new(0)),
    }
}

fn packed_head_fixture(recurrent: bool) -> Model {
    let mut model = fixture(recurrent);
    model.metadata.insert(
        "qwen35.embedding_length".into(),
        MetadataValue::from(256_u64),
    );
    for tensor in model.tensors.values_mut() {
        let name = tensor.name.as_str();
        let input = matches!(
            name,
            "token_embd.weight"
                | "output.weight"
                | "blk.0.ffn_up.weight"
                | "blk.0.ffn_gate.weight"
                | "blk.0.attn_q.weight"
                | "blk.0.attn_k.weight"
                | "blk.0.attn_v.weight"
                | "blk.0.attn_qkv.weight"
                | "blk.0.attn_gate.weight"
                | "blk.0.ssm_beta.weight"
                | "blk.0.ssm_alpha.weight"
        );
        if input {
            tensor.dimensions[0] = 256;
            tensor.kind = 142;
            let mut block = [0x55; 34];
            block[..2].copy_from_slice(&0x3c00_u16.to_le_bytes());
            tensor.data = Arc::from(block.repeat(2 * tensor.dimensions[1]));
        } else if matches!(
            name,
            "token_embd.weight"
                | "blk.0.attn_output.weight"
                | "blk.0.ssm_out.weight"
                | "blk.0.ffn_down.weight"
        ) {
            let axis = usize::from(name != "token_embd.weight");
            tensor.dimensions[axis] = 256;
            tensor.data = Arc::from(vec![0; tensor.dimensions.iter().product::<usize>() * 4]);
        } else if matches!(
            name,
            "output_norm.weight" | "blk.0.attn_norm.weight" | "blk.0.post_attention_norm.weight"
        ) {
            tensor.dimensions[0] = 256;
            tensor.data = Arc::from(1.0_f32.to_le_bytes().repeat(256));
        }
    }
    model
}

#[test]
fn projection_scope_admission_counts_attention_and_logits_but_excludes_embedding() -> Result<()> {
    for (recurrent, hidden) in [(false, 5), (true, 6)] {
        let mut bounded = executor(hidden, Some(1));
        let calls = bounded.calls.clone();
        let remaining = bounded.remaining.clone();
        let mut projected = Qwen35::new(packed_head_fixture(recurrent), 16)?
            .with_projection_executor(&mut bounded)?;
        assert!(projected.admit_work(1, 0).is_ok());
        assert!(projected.admit_work(0, 1).is_err());
        assert!(projected.evaluate(0).is_err());
        assert_eq!(calls.get(), 0);
        remaining.set(hidden + 1);
        projected.evaluate(0)?;
        assert_eq!(u64::try_from(calls.get())?, hidden + 1);
        assert_eq!(remaining.get(), 0);
    }
    Ok(())
}

#[test]
fn projection_scope_admission_refuses_whole_token_without_state_change_or_fault() -> Result<()> {
    for recurrent in [false, true] {
        let mut bounded = executor(3, Some(4));
        let calls = bounded.calls.clone();
        let remaining = bounded.remaining.clone();
        let decoder = Qwen35::new(backend_fixture(recurrent, 142, 256)?, 16)?;
        let before = graph_bits(&decoder)?;
        let mut projected = decoder.with_projection_executor(&mut bounded)?;
        let checkpoint = projected.checkpoint()?;
        assert!(projected.evaluate(0).is_err());
        assert!(projected.ingest(0).is_err());
        assert_eq!(calls.get(), 0);
        assert_eq!(remaining.get(), 3);
        assert_eq!(graph_bits(projected.decoder())?, before);
        assert!(!projected.is_faulted());
        projected.restore(&checkpoint)?;
        projected.reset()?;
    }
    Ok(())
}

#[test]
fn projection_scope_admission_requires_exact_nonzero_selected_costs_and_checks_overflow()
-> Result<()> {
    for cost in [None, Some(0)] {
        let mut bounded = executor(10, cost);
        assert!(
            Qwen35::new(backend_fixture(false, 142, 256)?, 16)?
                .with_projection_executor(&mut bounded)
                .is_err()
        );
        assert_eq!(bounded.calls.get(), 0);
    }
    let mut bounded = executor(u64::MAX, Some(u64::MAX));
    let projected = Qwen35::new(backend_fixture(false, 142, 256)?, 16)?
        .with_projection_executor(&mut bounded)?;
    assert!(projected.admit_work(0, 1).is_ok());
    assert!(projected.admit_work(1, 1).is_err());
    assert!(projected.admit_work(usize::MAX, usize::MAX).is_err());
    Ok(())
}

#[test]
fn projection_scope_admission_reserves_full_generation_before_first_forward() -> Result<()> {
    let mut bounded = executor(3, Some(1));
    let calls = bounded.calls.clone();
    let projected = Qwen35::new(backend_fixture(true, 142, 256)?, 16)?
        .with_projection_executor(&mut bounded)?;
    let mut runtime = Runtime::new(
        projected,
        Limits {
            context_tokens: 16,
            checkpoints: 0,
            checkpoint_bytes: 1 << 24,
        },
    )?;
    let options = Generation {
        max_tokens: 3,
        stop_tokens: vec![0, 1],
    };
    // Two prompt forwards plus all three reserved output forwards exceed the
    // budget, even though either sampled token would stop immediately.
    assert!(
        runtime
            .generate(&[0, 1], &options, &Cancellation::default(), |_| {
                Err(std::io::Error::other("refused request must not emit").into())
            })
            .is_err()
    );
    assert_eq!(calls.get(), 0);
    assert!(!runtime.decoder().is_faulted());
    let options = Generation {
        max_tokens: 1,
        stop_tokens: vec![],
    };
    runtime.generate(&[0, 1], &options, &Cancellation::default(), |_| Ok(()))?;
    assert_eq!(calls.get(), 3);
    assert!(
        runtime
            .generate(&[0], &options, &Cancellation::default(), |_| Ok(()))
            .is_err()
    );
    assert_eq!(calls.get(), 3);
    Ok(())
}

#[test]
fn projection_scope_admission_charges_only_unreused_prompt_and_never_restores_device_budget()
-> Result<()> {
    let mut bounded = executor(5, Some(1));
    let remaining = bounded.remaining.clone();
    let calls = bounded.calls.clone();
    let projected = Qwen35::new(backend_fixture(false, 142, 256)?, 16)?
        .with_projection_executor(&mut bounded)?;
    let mut runtime = Runtime::new(
        projected,
        Limits {
            context_tokens: 16,
            checkpoints: 0,
            checkpoint_bytes: 1 << 24,
        },
    )?;
    let options = Generation {
        max_tokens: 1,
        stop_tokens: vec![],
    };
    let mut prompt = vec![0, 1];
    let mut emitted = Vec::new();
    runtime.generate(&prompt, &options, &Cancellation::default(), |token| {
        emitted.push(token);
        Ok(())
    })?;
    prompt.extend(emitted);
    assert_eq!(calls.get(), 3);
    assert_eq!(remaining.get(), 2);
    prompt.push(1);
    let completion = runtime.generate(&prompt, &options, &Cancellation::default(), |_| Ok(()))?;
    assert_eq!(completion.reused_tokens, 3);
    assert_eq!(calls.get(), 5);
    assert_eq!(remaining.get(), 0);
    runtime.reset()?;
    assert_eq!(remaining.get(), 0);
    assert!(
        runtime
            .generate(&[0], &options, &Cancellation::default(), |_| Ok(()))
            .is_err()
    );
    assert_eq!(calls.get(), 5);
    Ok(())
}
