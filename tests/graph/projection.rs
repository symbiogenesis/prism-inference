//! Exercise real attention/recurrent graphs through the explicit executor seam.

use std::{cell::Cell, rc::Rc};

use bonsai_prism_runtime::kernels::Pq2ProjectionExecutor;
use bonsai_ternary::{Q8KBlock, half_to_float};

use super::*;

struct ReferenceExecutor {
    calls: Rc<Cell<usize>>,
    fail: Rc<Cell<bool>>,
    malformed: bool,
}

impl Pq2ProjectionExecutor for ReferenceExecutor {
    fn identity(&self) -> &'static str {
        "independent-scalar-pq2-control"
    }
    fn supports(&self, _: &Tensor) -> bool {
        true
    }
    #[allow(
        clippy::as_conversions,
        clippy::cast_precision_loss,
        clippy::suboptimal_flops,
        reason = "Exactly bounded integer fixture dot and declared source/fused F32 reduction."
    )]
    fn project(
        &mut self,
        tensor: &Tensor,
        activation: &[Q8KBlock],
        fused: bool,
        output: &mut Vec<f32>,
    ) -> Result<()> {
        self.calls.set(self.calls.get() + 1);
        output.clear();
        if self.fail.get() {
            return Err(std::io::Error::other("synthetic submitted device loss").into());
        }
        if self.malformed {
            output.push(42.0);
            return Ok(());
        }
        let blocks = tensor.dimensions[0] / 128;
        for row in tensor.data.chunks_exact(blocks * 34) {
            let mut sum = 0.0_f32;
            for (index, block) in row.as_chunks::<34>().0.iter().enumerate() {
                let prepared = &activation[index / 2];
                let mut dot = 0_i32;
                for lane in 0..128 {
                    let symbol = (block[2 + lane / 4] >> (lane % 4 * 2)) & 3;
                    dot += (i32::from(symbol) - 1)
                        * i32::from(prepared.values[index % 2 * 128 + lane]);
                }
                let factor =
                    half_to_float(u16::from_le_bytes([block[0], block[1]])) * prepared.scale;
                sum = if fused {
                    factor.mul_add(dot as f32, sum)
                } else {
                    sum + factor * dot as f32
                };
            }
            output.push(sum);
        }
        Ok(())
    }
}

fn executor() -> (ReferenceExecutor, Rc<Cell<usize>>, Rc<Cell<bool>>) {
    let calls = Rc::new(Cell::new(0));
    let fail = Rc::new(Cell::new(false));
    (
        ReferenceExecutor {
            calls: calls.clone(),
            fail: fail.clone(),
            malformed: false,
        },
        calls,
        fail,
    )
}

#[test]
fn projection_scope_preserves_logits_state_adapter_basis_checkpoint_and_reset() -> Result<()> {
    let bits = |values: Vec<f32>| values.into_iter().map(f32::to_bits).collect::<Vec<_>>();
    for recurrent in [false, true] {
        for adapter_scale in [0.0, 1.3] {
            let build = || -> Result<Qwen35> {
                let mut base = backend_fixture(recurrent, 142, 256)?;
                writer_transform(&mut base, "blk.0.ffn_down.weight", false);
                base.metadata.insert(
                    "prism.hadamard.sign_widths".into(),
                    MetadataValue::from(vec![256_u64]),
                );
                base.metadata.insert(
                    "prism.hadamard.sign_values".into(),
                    MetadataValue::from([-1_i64, 1].repeat(128)),
                );
                let target = if recurrent {
                    "blk.0.ssm_out.weight"
                } else {
                    "blk.0.attn_output.weight"
                };
                let factors = rank_one_factors(
                    &base,
                    &["token_embd.weight", "blk.0.ffn_down.weight", target],
                    0.125,
                );
                rank_one_graph(base, factors, adapter_scale)
            };
            let mut control = build()?;
            let (mut executor, calls, _) = executor();
            let mut graph = build()?.with_projection_executor(&mut executor)?;
            assert_ne!(graph.identity(), control.identity());
            for token in [0, 1, 0] {
                assert_eq!(bits(graph.evaluate(token)?), bits(control.evaluate(token)?));
                assert_eq!(graph_bits(graph.decoder())?, graph_bits(&control)?);
            }
            let checkpoint = graph.checkpoint()?;
            let control_checkpoint = control.checkpoint()?;
            graph.ingest(1)?;
            control.ingest(1)?;
            assert_eq!(graph_bits(graph.decoder())?, graph_bits(&control)?);
            graph.restore(&checkpoint)?;
            control.restore(&control_checkpoint)?;
            assert_eq!(bits(graph.evaluate(1)?), bits(control.evaluate(1)?));
            graph.reset()?;
            control.reset()?;
            assert_eq!(bits(graph.evaluate(0)?), bits(control.evaluate(0)?));
            assert_eq!(graph_bits(graph.decoder())?, graph_bits(&control)?);
            assert_eq!(calls.get(), 6);
        }
    }
    Ok(())
}

#[test]
fn projection_scope_device_loss_after_prepared_state_delta_never_replays_or_reopens() -> Result<()>
{
    for recurrent in [false, true] {
        let (mut executor, calls, fail) = executor();
        let mut graph = Qwen35::new(backend_fixture(recurrent, 142, 256)?, 16)?
            .with_projection_executor(&mut executor)?;
        graph.evaluate(0)?;
        let before = graph_bits(graph.decoder())?;
        let checkpoint = graph.checkpoint()?;
        fail.set(true);
        let mut output = vec![42.0];
        assert!(graph.evaluate_into(1, &mut output).is_err());
        assert_eq!(output, Vec::<f32>::new());
        assert_eq!(graph_bits(graph.decoder())?, before);
        assert!(graph.is_faulted());
        assert!(graph.evaluate(1).is_err());
        assert!(graph.ingest(1).is_err());
        assert!(graph.reset().is_err());
        assert!(graph.restore(&checkpoint).is_err());
        assert_eq!(calls.get(), 2);
    }
    Ok(())
}

#[test]
fn projection_scope_keeps_q8_zero_and_ptq1_paths_on_cpu_and_checks_complete_output() -> Result<()> {
    for (kind, width) in [(142, 128), (143, 256)] {
        let (mut executor, calls, _) = executor();
        let mut graph = Qwen35::new(backend_fixture(false, kind, width)?, 16)?
            .with_projection_executor(&mut executor)?;
        graph.evaluate(0)?;
        assert_eq!(calls.get(), 0);
    }
    let (mut executor, calls, _) = executor();
    executor.malformed = true;
    let mut graph = Qwen35::new(backend_fixture(false, 142, 256)?, 16)?
        .with_projection_executor(&mut executor)?;
    let mut output = vec![42.0];
    assert!(graph.evaluate_into(0, &mut output).is_err());
    assert_eq!(output, Vec::<f32>::new());
    assert_eq!(calls.get(), 1);
    assert!(graph.is_faulted());
    Ok(())
}

#[test]
fn projection_scope_restores_its_original_cpu_state_and_rejects_foreign_graph() -> Result<()> {
    let mut decoder = Qwen35::new(backend_fixture(true, 142, 256)?, 16)?;
    decoder.evaluate(0)?;
    let before = graph_bits(&decoder)?;
    let original = decoder.checkpoint()?;
    let foreign = Qwen35::new(backend_fixture(true, 142, 256)?, 16)?.checkpoint()?;
    let (mut executor, _, _) = executor();
    let mut graph = decoder.with_projection_executor(&mut executor)?;
    graph.evaluate(1)?;
    graph.restore(&original)?;
    assert_eq!(graph_bits(graph.decoder())?, before);
    assert!(graph.restore(&foreign).is_err());
    assert_eq!(graph_bits(graph.decoder())?, before);
    assert!(!graph.is_faulted());
    Ok(())
}
