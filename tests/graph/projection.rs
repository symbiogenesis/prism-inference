//! Exercise real attention/recurrent graphs through the explicit executor seam.

use std::{cell::Cell, rc::Rc};

use bonsai_prism_runtime::kernels::{Pq2ProjectionExecutor, Q8Block};
use bonsai_ternary::{Q8KBlock, decode_ptq_trits, half_to_float};

use super::*;

struct ReferenceExecutor {
    calls: Rc<Cell<usize>>,
    fail: Rc<Cell<bool>>,
    malformed: bool,
    q8: bool,
    f32_direct: bool,
}

impl Pq2ProjectionExecutor for ReferenceExecutor {
    fn identity(&self) -> &'static str {
        "independent-scalar-pq2-control"
    }
    fn supports(&self, _: &Tensor) -> bool {
        true
    }

    fn supports_q8(&self, _: &Tensor) -> bool {
        self.q8
    }

    fn supports_f32(&self, tensor: &Tensor) -> bool {
        self.f32_direct && tensor.kind == 142
    }

    fn project_f32(
        &mut self,
        tensor: &Tensor,
        activation: &[f32],
        _fused: bool,
        output: &mut Vec<f32>,
    ) -> Result<()> {
        self.calls.set(self.calls.get() + 1);
        output.clear();
        if self.fail.get() {
            return Err(std::io::Error::other("synthetic submitted F32 loss").into());
        }
        let mut scratch =
            bonsai_prism_runtime::kernels::MatvecScratch::with_backend(MatvecBackend::Scalar);
        scratch
            .prepare(tensor, activation)?
            .matvec_into(tensor, output)
    }

    #[allow(
        clippy::as_conversions,
        clippy::cast_precision_loss,
        clippy::suboptimal_flops,
        reason = "Independent scalar original-code reduction with exact bounded integer sums."
    )]
    fn project_q8(
        &mut self,
        tensor: &Tensor,
        activation: &[Q8Block],
        pinned: bool,
        output: &mut Vec<f32>,
    ) -> Result<()> {
        self.calls.set(self.calls.get() + 1);
        output.clear();
        if self.fail.get() {
            return Err(std::io::Error::other("synthetic submitted Q8_0 loss").into());
        }
        let bytes = if tensor.kind == 143 { 28 } else { 34 };
        let blocks = tensor.dimensions[0] / 128;
        for row in tensor.data.chunks_exact(blocks * bytes) {
            let mut lanes = [0.0_f32; 4];
            let mut sum = 0.0_f32;
            for (index, block) in row.chunks_exact(bytes).enumerate() {
                let (weights, d) = if tensor.kind == 143 {
                    (
                        decode_ptq_trits(block.try_into()?),
                        half_to_float(u16::from_le_bytes([block[26], block[27]])),
                    )
                } else {
                    (
                        std::array::from_fn(|lane| {
                            i16::from((block[2 + lane / 4] >> (lane % 4 * 2)) & 3) - 1
                        }),
                        half_to_float(u16::from_le_bytes([block[0], block[1]])),
                    )
                };
                let mut block_sum = 0.0_f32;
                for group in 0..4 {
                    let activation = &activation[index * 4 + group];
                    let mut integers = [0_i32; 4];
                    for (lane, integer) in integers.iter_mut().enumerate() {
                        for start in [lane * 4, lane * 4 + 16] {
                            for position in start..start + 4 {
                                *integer += i32::from(weights[group * 32 + position])
                                    * i32::from(activation.values[position]);
                            }
                        }
                    }
                    if pinned && tensor.kind == 143 {
                        for (lane, integer) in lanes.iter_mut().zip(integers) {
                            *lane = (integer as f32).mul_add(d * activation.scale, *lane);
                        }
                    } else {
                        block_sum += activation.scale * integers.into_iter().sum::<i32>() as f32;
                    }
                }
                if !pinned || tensor.kind != 143 {
                    sum += d * block_sum;
                }
            }
            output.push(if pinned && tensor.kind == 143 {
                (lanes[0] + lanes[1]) + (lanes[2] + lanes[3])
            } else {
                sum
            });
        }
        Ok(())
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
            q8: false,
            f32_direct: false,
        },
        calls,
        fail,
    )
}

#[test]
fn projection_scope_q8_opt_in_preserves_ptq1_odd_pq2_state_and_adapter() -> Result<()> {
    let bits = |values: Vec<f32>| values.into_iter().map(f32::to_bits).collect::<Vec<_>>();
    for (kind, width) in [(142, 128), (143, 256)] {
        for recurrent in [false, true] {
            for adapter_scale in [0.0, 1.3] {
                let build = || -> Result<Qwen35> {
                    let base = backend_fixture(recurrent, kind, width)?;
                    let target = if recurrent {
                        "blk.0.ssm_out.weight"
                    } else {
                        "blk.0.attn_output.weight"
                    };
                    let factors =
                        rank_one_factors(&base, &["blk.0.ffn_down.weight", target], 0.125);
                    rank_one_graph(base, factors, adapter_scale)
                };
                let mut control = build()?;
                let (mut executor, calls, fail) = executor();
                executor.q8 = true;
                let mut graph = build()?.with_projection_executor(&mut executor)?;
                for token in [0, 1, 0] {
                    assert_eq!(bits(graph.evaluate(token)?), bits(control.evaluate(token)?));
                    assert_eq!(graph_bits(graph.decoder())?, graph_bits(&control)?);
                }
                assert_eq!(calls.get(), 3);
                let before = graph_bits(graph.decoder())?;
                fail.set(true);
                let mut output = vec![42.0];
                assert!(graph.evaluate_into(1, &mut output).is_err());
                assert_eq!(output, Vec::<f32>::new());
                assert!(graph.is_faulted());
                assert_eq!(graph_bits(graph.decoder())?, before);
                assert!(graph.evaluate(1).is_err());
                assert_eq!(calls.get(), 4);
            }
        }
    }
    Ok(())
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

#[test]
fn tiled_prefill_projection_scope_dispatches_pq2_projections_and_preserves_state_bits() -> Result<()>
{
    let prompt = [0, 1, 1, 0, 1, 0, 0, 1, 0];
    for recurrent in [false, true] {
        for f32_direct in [false, true] {
            let build = || -> Result<Qwen35> {
                let model = backend_fixture(recurrent, 142, 256)?;
                let factors = rank_one_factors(&model, &["blk.0.ffn_down.weight"], 0.125);
                rank_one_graph(model, factors, 1.3)
            };
            let mut expected = build()?;
            for token in prompt {
                expected.ingest(token)?;
            }
            let base = build()?;
            let reserve = base.tiled_prefill_scratch_bytes(8)?;
            let tiled = base.with_tiled_prefill(8, reserve)?;
            let (mut executor, calls, _) = executor();
            executor.f32_direct = f32_direct;
            let mut graph = tiled.with_projection_executor(&mut executor)?;
            graph.ingest_block(&prompt, &|| Ok(()))?;
            assert_eq!(calls.get(), prompt.len());
            assert_eq!(graph_bits(graph.decoder())?, graph_bits(&expected)?);
            assert_eq!(graph.evaluate(1)?, expected.evaluate(1)?);
        }
    }
    Ok(())
}
