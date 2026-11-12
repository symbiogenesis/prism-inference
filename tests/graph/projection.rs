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

    #[expect(
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
    #[expect(
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
            let mut output = Vec::new();
            assert_ne!(graph.identity(), control.identity());
            for token in [0, 1, 0] {
                let view = graph.evaluate_finite_into(token, &mut output)?;
                assert_eq!(
                    bits(view.as_slice().to_vec()),
                    bits(control.evaluate(token)?)
                );
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
fn lut_projection_scope_preserves_full_and_recurrent_state_with_adapter() -> Result<()> {
    use bonsai_prism_runtime::kernels::{Pq2LutExecutor, RowParallelism};

    let bits = |values: Vec<f32>| values.into_iter().map(f32::to_bits).collect::<Vec<_>>();
    for width in [128, 256] {
        for recurrent in [false, true] {
            let build = || -> Result<Qwen35> {
                let base = backend_fixture(recurrent, 142, width)?;
                let factors = rank_one_factors(&base, &["blk.0.ffn_down.weight"], 0.125);
                rank_one_graph(base, factors, 1.3)
            };
            let mut control = build()?;
            let mut executor = Pq2LutExecutor::new(256, RowParallelism::new(1)?)?;
            let mut graph = build()?.with_projection_executor(&mut executor)?;
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
        assert!(graph.evaluate_finite_into(1, &mut output).is_err());
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
    assert!(graph.evaluate_finite_into(0, &mut output).is_err());
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

#[cfg(feature = "parallel-rows")]
fn sibling_graph_weight(name: &str, rows: usize) -> Tensor {
    let mut data = Vec::new();
    for row in 0..rows {
        let mut block = [if row % 2 == 0 { 0x66 } else { 0x99 }; 34];
        block[..2].copy_from_slice(&0x2000_u16.to_le_bytes());
        data.extend(block.repeat(2));
    }
    Tensor {
        name: name.to_owned(),
        dimensions: vec![256, rows],
        kind: 142,
        data: Arc::from(data),
    }
}

#[cfg(feature = "parallel-rows")]
fn sibling_graph_fixture(recurrent: bool) -> Result<Model> {
    let mut base = backend_fixture(recurrent, 142, 256)?;
    base.metadata.insert(
        "qwen35.embedding_length".to_owned(),
        MetadataValue::from(256_u64),
    );
    // Keep the fixture's two-element KV/GDN heads, but widen the residual
    // stream so real sibling projections consume complete Q8_K blocks.
    for tensor in base.tensors.values_mut() {
        match tensor.name.as_str() {
            "token_embd.weight" | "output.weight" => {
                tensor.dimensions[0] = 256;
                let data: Vec<_> = tensor
                    .data
                    .as_chunks::<8>()
                    .0
                    .iter()
                    .flat_map(|row| row.repeat(128))
                    .collect();
                tensor.data = Arc::from(data);
            }
            "output_norm.weight"
            | "blk.0.attn_norm.weight"
            | "blk.0.post_attention_norm.weight" => {
                tensor.dimensions[0] = 256;
                tensor.data = Arc::from(tensor.data.repeat(128));
            }
            "blk.0.ffn_up.weight"
            | "blk.0.ffn_gate.weight"
            | "blk.0.attn_q.weight"
            | "blk.0.attn_k.weight"
            | "blk.0.attn_v.weight"
            | "blk.0.attn_qkv.weight"
            | "blk.0.attn_gate.weight" => {
                *tensor = sibling_graph_weight(&tensor.name, tensor.dimensions[1]);
            }
            "blk.0.ffn_down.weight" | "blk.0.attn_output.weight" | "blk.0.ssm_out.weight" => {
                tensor.dimensions[1] = 256;
                tensor.data = Arc::from(tensor.data.repeat(128));
            }
            "blk.0.ssm_beta.weight" | "blk.0.ssm_alpha.weight" => {
                tensor.dimensions[0] = 256;
                tensor.data = Arc::from(vec![0; 256 * 4]);
            }
            _ => {}
        }
    }
    writer_transform(&mut base, "blk.0.ffn_down.weight", false);
    base.metadata.insert(
        "prism.hadamard.sign_widths".to_owned(),
        MetadataValue::from(vec![256_u64]),
    );
    base.metadata.insert(
        "prism.hadamard.sign_values".to_owned(),
        MetadataValue::from([-1_i64, 1].repeat(128)),
    );
    Ok(base)
}

#[cfg(feature = "parallel-rows")]
#[test]
fn cpu_sibling_projection_preserves_full_and_recurrent_state_with_adapter() -> Result<()> {
    use bonsai_prism_runtime::kernels::RowParallelism;

    let bits = |values: Vec<f32>| values.into_iter().map(f32::to_bits).collect::<Vec<_>>();
    for recurrent in [false, true] {
        for adapter_scale in [0.0, 1.3] {
            let build = || -> Result<Qwen35> {
                let base = sibling_graph_fixture(recurrent)?;
                let output = if recurrent {
                    "blk.0.ssm_out.weight"
                } else {
                    "blk.0.attn_output.weight"
                };
                let factors = rank_one_factors(&base, &["blk.0.ffn_down.weight", output], 0.125);
                rank_one_graph(base, factors, adapter_scale)
            };
            let mut control = build()?;
            let mut graph = build()?;
            graph.set_row_parallelism(RowParallelism::new(4)?);
            for token in [0, 1, 0] {
                assert_eq!(bits(graph.evaluate(token)?), bits(control.evaluate(token)?));
                assert_eq!(graph_bits(&graph)?, graph_bits(&control)?);
            }
            assert!(graph.backend_statistics().grouped_row_dispatches > 0);
            let checkpoint = graph.checkpoint()?;
            let control_checkpoint = control.checkpoint()?;
            graph.ingest(1)?;
            control.ingest(1)?;
            assert_eq!(graph_bits(&graph)?, graph_bits(&control)?);
            graph.restore(&checkpoint)?;
            control.restore(&control_checkpoint)?;
            assert_eq!(bits(graph.evaluate(1)?), bits(control.evaluate(1)?));
            let before_failure = graph_bits(&graph)?;
            assert!(graph.evaluate(2).is_err());
            assert_eq!(graph_bits(&graph)?, before_failure);
            graph.reset()?;
            control.reset()?;
            assert_eq!(bits(graph.evaluate(0)?), bits(control.evaluate(0)?));
            assert_eq!(graph_bits(&graph)?, graph_bits(&control)?);
        }
    }
    Ok(())
}
