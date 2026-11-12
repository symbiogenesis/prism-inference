use super::*;
use hexagon_native_rpc::{Error as NativeError, NativeQ2WeightRegistry};
use std::time::Duration;

struct TileSubmitter {
    registry: NativeQ2WeightRegistry,
    memory: Vec<u8>,
    weights: Vec<Vec<u8>>,
    activations: Vec<Vec<f32>>,
    resident_calls: usize,
    fail_at: Option<usize>,
    failure: u8,
    batch_limit: usize,
    packets: Vec<usize>,
}

impl TileSubmitter {
    #[expect(
        clippy::suboptimal_flops,
        reason = "The independent fake DSP retains explicit scalar multiply/add order"
    )]
    fn compute(
        &mut self,
        columns: usize,
        rows: usize,
        weights: &[u8],
        activation: &[f32],
        output: &mut [f32],
    ) -> std::result::Result<(), NativeError> {
        assert_eq!(activation.len(), columns);
        assert_eq!(output.len(), rows);
        self.weights.push(weights.to_vec());
        self.activations.push(activation.to_vec());
        let fail = self.fail_at == Some(self.activations.len());
        if fail && self.failure == 0 {
            output.fill(123.0);
            return Err(NativeError::Contract);
        }
        for (row, destination) in output.iter_mut().enumerate() {
            let mut sum = 0.0_f32;
            for block in 0..columns / 128 {
                let pair = (row / 32 * (columns / 128) + block) * 1152;
                let scale_at = pair + 512 + (row % 32) * 2;
                let factor = bonsai_ternary::half_to_float(u16::from_le_bytes([
                    weights[scale_at],
                    weights[scale_at + 1],
                ]));
                let mut dot = 0.0_f32;
                for index in 0..128 {
                    let group = (index % 32) / 4;
                    let lane = index % 4;
                    let at =
                        pair + (group / 4) * 576 + (group % 4) * 128 + (row % 32) * 4 + index / 32;
                    let code = (weights[at] >> (lane * 2)) & 3;
                    dot += f32::from(i16::from(code) - 1) * activation[block * 128 + index];
                }
                sum += factor * dot;
            }
            *destination = sum;
        }
        if fail && self.failure == 1 {
            output[0] = f32::INFINITY;
        } else if fail && self.failure == 2 {
            output[rows - 1] = 1.0;
        }
        Ok(())
    }

    fn batch<const B: usize>(
        &mut self,
        handles: [&NativeQ2WeightHandle; B],
        activations: [&[f32]; B],
        outputs: [&mut [f32]; B],
    ) -> std::result::Result<(), NativeError> {
        self.packets.push(B);
        for ((handle, activation), output) in handles.into_iter().zip(activations).zip(outputs) {
            self.submit_resident(handle, activation, output)?;
        }
        Ok(())
    }
}

impl NativeQ2Submitter for TileSubmitter {
    fn maximum_resident_token_batch(&self) -> usize {
        self.batch_limit
    }

    fn submit_resident_multi2(
        &mut self,
        handles: [&NativeQ2WeightHandle; 2],
        activations: [&[f32]; 2],
        outputs: [&mut [f32]; 2],
    ) -> std::result::Result<(), NativeError> {
        self.batch(handles, activations, outputs)
    }

    fn submit_resident_multi3(
        &mut self,
        handles: [&NativeQ2WeightHandle; 3],
        activations: [&[f32]; 3],
        outputs: [&mut [f32]; 3],
    ) -> std::result::Result<(), NativeError> {
        self.batch(handles, activations, outputs)
    }

    fn submit_resident_multi4(
        &mut self,
        handles: [&NativeQ2WeightHandle; 4],
        activations: [&[f32]; 4],
        outputs: [&mut [f32]; 4],
    ) -> std::result::Result<(), NativeError> {
        self.batch(handles, activations, outputs)
    }

    fn submit_resident_multi8(
        &mut self,
        handles: [&NativeQ2WeightHandle; 8],
        activations: [&[f32]; 8],
        outputs: [&mut [f32]; 8],
    ) -> std::result::Result<(), NativeError> {
        self.batch(handles, activations, outputs)
    }

    fn residency_capacity_bytes(&self) -> usize {
        self.registry.remaining_bytes()
    }

    fn prepare_weights(
        &mut self,
        columns: usize,
        rows: usize,
        weights: &[u8],
    ) -> std::result::Result<Option<NativeQ2WeightHandle>, NativeError> {
        let handle = self.registry.prepare(columns, rows, weights)?;
        if let Some(handle) = &handle {
            self.memory[self.registry.range(handle)?].copy_from_slice(weights);
        }
        Ok(handle)
    }

    fn submit_resident(
        &mut self,
        handle: &NativeQ2WeightHandle,
        activation: &[f32],
        output: &mut [f32],
    ) -> std::result::Result<(), NativeError> {
        let weights = self.memory[self.registry.range(handle)?].to_vec();
        self.resident_calls += 1;
        self.compute(
            handle.columns(),
            handle.rows(),
            &weights,
            activation,
            output,
        )
    }

    fn submit(
        &mut self,
        columns: usize,
        rows: usize,
        weights: &[u8],
        activation: &[f32],
        output: &mut [f32],
    ) -> std::result::Result<(), NativeError> {
        self.compute(columns, rows, weights, activation, output)
    }
}

fn tensor(rows: usize) -> Tensor {
    let columns = 384;
    let mut data = vec![0_u8; rows * 3 * 34];
    for (index, block) in data.as_chunks_mut::<34>().0.iter_mut().enumerate() {
        let bits = [0x3800_u16, 0x3c00, 0x4000][index % 3];
        block[..2].copy_from_slice(&bits.to_le_bytes());
        for (lane, byte) in block[2..].iter_mut().enumerate() {
            *byte = u8::try_from((index * 13 + lane * 7) % 256).unwrap_or(0);
        }
    }
    Tensor {
        name: "blk.0.ffn_down.weight".to_owned(),
        dimensions: vec![columns, rows],
        kind: 142,
        data: Arc::from(data),
    }
}

fn executor(
    tensor: &Tensor,
    max_columns: usize,
    resident: bool,
) -> Result<NativeQ2Executor<TileSubmitter>> {
    let metadata = if resident { 8192 } else { 0 };
    let submitter = TileSubmitter {
        registry: NativeQ2WeightRegistry::new(65_536)?,
        memory: vec![0; 65_536],
        weights: Vec::new(),
        activations: Vec::new(),
        resident_calls: 0,
        fail_at: None,
        failure: 0,
        batch_limit: 1,
        packets: Vec::new(),
    };
    let limits = NativeQ2ProjectionLimits::new(max_columns, 32, 10_000, Duration::from_secs(10))?;
    let payload = tensor.dimensions[1] * 4 + 32 * 36 * (max_columns / 128) + 32 * 4 + metadata;
    let executor = NativeQ2Executor::new(submitter, limits, payload);
    if resident {
        executor.with_resident_weights(metadata)
    } else {
        Ok(executor)
    }
}

fn activations(tokens: usize) -> Vec<Vec<f32>> {
    (0..tokens)
        .map(|token| {
            (0..384)
                .map(|index| {
                    f32::from(i16::try_from((token * 3 + index) % 17).unwrap_or(0) - 8) * 0.125
                })
                .collect()
        })
        .collect()
}

#[expect(
    clippy::suboptimal_flops,
    reason = "Independent original-PQ2 oracle retains block and tile accumulation order"
)]
fn oracle(tensor: &Tensor, activation: &[f32], max_columns: usize) -> Vec<f32> {
    (0..tensor.dimensions[1])
        .map(|row| {
            let mut sum = 0.0_f32;
            for start in (0..384).step_by(max_columns) {
                let end = (start + max_columns).min(384);
                let mut tile = 0.0_f32;
                for block in start / 128..end / 128 {
                    let original = &tensor.data[(row * 3 + block) * 34..][..34];
                    let factor = bonsai_ternary::half_to_float(u16::from_le_bytes([
                        original[0],
                        original[1],
                    ]));
                    let mut dot = 0.0_f32;
                    for index in 0..128 {
                        let code = (original[2 + index % 32] >> (2 * (index / 32))) & 3;
                        dot += f32::from(i16::from(code) - 1) * activation[block * 128 + index];
                    }
                    tile += factor * dot;
                }
                sum += tile;
            }
            sum
        })
        .collect()
}

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "Keep numerical, allocation, traversal, cached-repeat and admission assertions together"
)]
fn native_q2_multitile_tokens_match_scalar_transient_and_resident() -> Result<()> {
    for rows in [49, 65] {
        let tensor = tensor(rows);
        for max_columns in [128, 256] {
            for resident in [false, true] {
                for tokens in [0, 1, 8, 16, 32] {
                    let mut executor = executor(&tensor, max_columns, resident)?;
                    let activations = activations(tokens);
                    let mut outputs = vec![vec![91.0; rows]; tokens];
                    let addresses: Vec<_> = outputs.iter().map(Vec::as_ptr).collect();
                    executor.project_f32_tokens(&tensor, &activations, true, &mut outputs)?;
                    for (token, (activation, output)) in
                        activations.iter().zip(&outputs).enumerate()
                    {
                        assert_eq!(output.as_ptr(), addresses[token]);
                        let expected = oracle(&tensor, activation, max_columns);
                        assert_eq!(
                            output
                                .iter()
                                .map(|value| value.to_bits())
                                .collect::<Vec<_>>(),
                            expected
                                .iter()
                                .map(|value| value.to_bits())
                                .collect::<Vec<_>>()
                        );
                    }
                    let tiles = rows.div_ceil(32) * 384_usize.div_ceil(max_columns);
                    let calls = tiles * tokens;
                    assert_eq!(executor.submitter.activations.len(), calls);
                    assert_eq!(
                        executor.statistics().operations_completed,
                        u64::try_from(calls)?
                    );
                    assert_eq!(
                        executor.statistics().projections_completed,
                        u64::try_from(tokens)?
                    );
                    assert_eq!(executor.queue_writes_completed(), u64::try_from(calls)?);
                    assert_eq!(
                        executor.submitter.resident_calls,
                        if resident { calls } else { 0 }
                    );
                    if tokens != 0 {
                        for tile_weights in executor.submitter.weights.chunks_exact(tokens) {
                            assert!(
                                tile_weights
                                    .iter()
                                    .all(|weights| weights == &tile_weights[0])
                            );
                        }
                        let column_tiles = 384_usize.div_ceil(max_columns);
                        for (tile, token_activations) in executor
                            .submitter
                            .activations
                            .chunks_exact(tokens)
                            .enumerate()
                        {
                            let start = (tile % column_tiles) * max_columns;
                            let end = (start + max_columns).min(384);
                            for (actual, original) in token_activations.iter().zip(&activations) {
                                assert_eq!(actual, &original[start..end]);
                            }
                        }
                        let prepared = executor.residency_statistics();
                        assert_eq!(prepared.tensors_prepared, u64::from(resident));
                        assert_eq!(
                            prepared.cache_hits,
                            if resident {
                                u64::try_from(tokens - 1)?
                            } else {
                                0
                            }
                        );
                        if resident {
                            let used = executor.submitter.registry.used_bytes();
                            executor.project_f32_tokens(
                                &tensor,
                                &activations,
                                true,
                                &mut outputs,
                            )?;
                            assert_eq!(executor.submitter.registry.used_bytes(), used);
                            assert_eq!(executor.residency_statistics().tensors_prepared, 1);
                            assert_eq!(
                                executor.residency_statistics().cache_hits,
                                u64::try_from(2 * tokens - 1)?
                            );
                            assert_eq!(
                                executor.statistics().operations_completed,
                                u64::try_from(2 * calls)?
                            );
                        }
                    }
                    let calls_before = executor.submitter.activations.len();
                    executor.maximum_owned_payload_bytes -= 1;
                    assert!(!executor.supports_f32(&tensor));
                    assert_eq!(executor.submitter.activations.len(), calls_before);
                }
            }
        }
    }
    Ok(())
}

#[test]
fn native_q2_multitile_tokens_preserve_failure_and_publication() -> Result<()> {
    let tensor = tensor(49);
    let activations = activations(3);
    for resident in [false, true] {
        for (failure, fail_at) in [(0, 1), (0, 12), (1, 2), (1, 12), (2, 12)] {
            let mut executor = executor(&tensor, 256, resident)?;
            executor.submitter.failure = failure;
            executor.submitter.fail_at = Some(fail_at);
            let mut outputs = vec![vec![91.0; 49]; 3];
            assert!(
                executor
                    .project_f32_tokens(&tensor, &activations, true, &mut outputs)
                    .is_err()
            );
            assert!(outputs.iter().all(Vec::is_empty));
            assert!(executor.is_faulted());
            assert_eq!(executor.submitter.activations.len(), fail_at);
            assert_eq!(
                executor.statistics().operations_attempted,
                u64::try_from(fail_at)?
            );
            assert_eq!(
                executor.statistics().operations_completed,
                u64::try_from(fail_at - 1)?
            );
            assert_eq!(
                executor.queue_writes_completed(),
                u64::try_from(fail_at - 1)?
            );
            assert_eq!(
                executor.statistics().projections_completed,
                u64::try_from(fail_at.saturating_sub(10))?
            );
            assert!(
                executor
                    .project_f32_tokens(&tensor, &activations, true, &mut outputs)
                    .is_err()
            );
            assert_eq!(executor.submitter.activations.len(), fail_at);
        }
    }
    Ok(())
}

#[test]
fn native_q2_multitile_tokens_reject_invalid_input_before_submission() -> Result<()> {
    for resident in [false, true] {
        for fault in 0..4 {
            let mut tensor = tensor(49);
            let mut activations = activations(3);
            if fault < 2 {
                activations[2][0] = f32::NAN;
            } else if fault == 2 {
                assert!(activations[2].pop().is_some());
            }
            if fault == 1 || fault == 3 {
                let mut data = tensor.data.to_vec();
                data[..2].copy_from_slice(&0x7e00_u16.to_le_bytes());
                tensor.data = Arc::from(data);
            }
            let mut executor = executor(&tensor, 256, resident)?;
            let mut outputs = vec![vec![91.0]; 3];
            assert!(
                executor
                    .project_f32_tokens(&tensor, &activations, true, &mut outputs)
                    .is_err()
            );
            assert!(outputs.iter().all(Vec::is_empty));
            assert!(executor.is_faulted());
            assert_eq!(executor.statistics().operations_attempted, 0);
            assert_eq!(executor.residency_statistics().tensors_prepared, 0);
            let no_activations: [Vec<f32>; 0] = [];
            assert_eq!(executor.submitter.activations, no_activations);
        }
    }
    Ok(())
}

#[test]
fn native_q2_resident_multitile_batches_match_original_tiles_and_queue_counts() -> Result<()> {
    let tensor = tensor(49);
    for batch_limit in [1, 4, 8] {
        for tokens in [2, 3, 5, 7, 8, 9, 16, 32] {
            let mut executor = executor(&tensor, 256, true)?;
            executor.submitter.batch_limit = batch_limit;
            let activations = activations(tokens);
            let mut outputs = vec![vec![91.0; 49]; tokens];
            let addresses: Vec<_> = outputs.iter().map(Vec::as_ptr).collect();
            executor.project_f32_tokens(&tensor, &activations, true, &mut outputs)?;
            for (token, (activation, output)) in activations.iter().zip(&outputs).enumerate() {
                assert_eq!(output.as_ptr(), addresses[token]);
                assert_eq!(
                    output
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>(),
                    oracle(&tensor, activation, 256)
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>()
                );
            }
            // Independent packet-count oracle: 8-op groups, then a <=4 tail.
            let groups = if batch_limit == 8 {
                tokens / 8 + (tokens % 8).div_ceil(4)
            } else {
                tokens.div_ceil(batch_limit)
            };
            assert_eq!(
                executor.queue_writes_completed(),
                u64::try_from(4 * groups)?
            );
            assert_eq!(
                executor.queue_writes_attempted(),
                u64::try_from(4 * groups)?
            );
            assert_eq!(
                executor.statistics().operations_completed,
                u64::try_from(4 * tokens)?
            );
            assert_eq!(
                executor.statistics().projections_completed,
                u64::try_from(tokens)?
            );
            assert_eq!(executor.submitter.resident_calls, 4 * tokens);
            assert_eq!(
                executor.submitter.packets.len(),
                if batch_limit == 1 {
                    0
                } else {
                    4 * groups - 4 * usize::from(tokens % 4 == 1)
                }
            );
            for (tile, actual) in executor
                .submitter
                .activations
                .chunks_exact(tokens)
                .enumerate()
            {
                let start = (tile % 2) * 256;
                let end = (start + 256).min(384);
                for (row, original) in actual.iter().zip(&activations) {
                    assert_eq!(row, &original[start..end]);
                }
            }
            let used = executor.submitter.registry.used_bytes();
            executor.project_f32_tokens(&tensor, &activations, true, &mut outputs)?;
            assert_eq!(executor.submitter.registry.used_bytes(), used);
            assert_eq!(
                executor.residency_statistics().cache_hits,
                u64::try_from(2 * tokens - 1)?
            );
            assert_eq!(
                executor.queue_writes_completed(),
                u64::try_from(8 * groups)?
            );
        }
    }
    Ok(())
}

#[test]
fn native_q2_resident_multitile_batch_failures_publish_nothing_and_never_replay() -> Result<()> {
    let tensor = tensor(49);
    let activations = activations(9);
    // Four weight tiles, each dispatched as 8 + 1. Probe failures within the
    // first and final packets, including the final partially padded row tile.
    for (failure, fail_at) in [(0, 3), (0, 30), (1, 3), (1, 30), (2, 30), (2, 36)] {
        let mut executor = executor(&tensor, 256, true)?;
        executor.submitter.batch_limit = 8;
        executor.submitter.failure = failure;
        executor.submitter.fail_at = Some(fail_at);
        let mut outputs = vec![vec![91.0; 49]; 9];
        assert!(
            executor
                .project_f32_tokens(&tensor, &activations, true, &mut outputs)
                .is_err()
        );
        assert!(outputs.iter().all(Vec::is_empty));
        assert!(executor.is_faulted());
        let completed_packets = if fail_at <= 8 {
            0
        } else if fail_at <= 35 {
            6
        } else {
            7
        };
        assert_eq!(executor.queue_writes_attempted(), completed_packets + 1);
        assert_eq!(executor.queue_writes_completed(), completed_packets);
        assert_eq!(
            executor.statistics().operations_attempted,
            if fail_at <= 8 {
                8
            } else if fail_at <= 35 {
                35
            } else {
                36
            }
        );
        assert_eq!(
            executor.statistics().operations_completed,
            if fail_at <= 8 {
                0
            } else if fail_at <= 35 {
                27
            } else {
                35
            }
        );
        let calls = executor.submitter.activations.len();
        assert!(
            executor
                .project_f32_tokens(&tensor, &activations, true, &mut outputs)
                .is_err()
        );
        assert_eq!(executor.submitter.activations.len(), calls);
    }
    Ok(())
}

#[test]
fn native_q2_resident_multitile_normalizes_capabilities_and_small_groups() -> Result<()> {
    let tensor = tensor(49);
    let activations = activations(3);
    for (batch_limit, groups) in [(0, 3), (2, 2), (3, 1), (5, 3), (7, 3), (9, 3)] {
        let mut executor = executor(&tensor, 256, true)?;
        executor.submitter.batch_limit = batch_limit;
        let mut outputs = vec![Vec::new(); 3];
        executor.project_f32_tokens(&tensor, &activations, true, &mut outputs)?;
        for (output, activation) in outputs.iter().zip(&activations) {
            assert_eq!(
                output
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                oracle(&tensor, activation, 256)
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>()
            );
        }
        assert_eq!(executor.queue_writes_completed(), 4 * groups);
        assert_eq!(executor.statistics().operations_completed, 12);
        assert_eq!(executor.statistics().projections_completed, 3);
    }
    Ok(())
}
