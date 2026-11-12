//! Production six-head tiled-attention dispatch and cancellation controls.

use std::cell::Cell;
use std::sync::Arc;

use super::*;

struct Harness {
    geometry: Geometry,
    weights: FullAttention,
    keys: KvCache,
    values: KvCache,
    projection: super::super::ProjectionScratch,
    transforms: Transforms,
    adapter_scratch: AdapterScratch,
    attention_scratch: super::super::AttentionScratch,
    tile: TiledPrefill,
    base: usize,
    count: usize,
}

fn sample(index: usize) -> f32 {
    [0.0, -0.0, 0.33337, -0.251, 0.501, -1.001, 2.001][index % 7]
}

impl Harness {
    fn new(count: usize, workers: usize) -> Result<Self> {
        // Use the real pinned rotary geometry; narrow four-channel fixtures
        // cannot execute pinned RMSNorm/RoPE in either comparison arm.
        let geometry = Geometry {
            embedding: 256,
            feed_forward: 256,
            heads: 24,
            kv_heads: 4,
            head: 256,
            rope: 64,
            rope_sections: [11, 11, 10, 0],
            rope_base: 10_000_000.0,
            epsilon: 1e-6,
            ssm_head: 128,
            ssm_keys: 16,
            ssm_values: 48,
            convolution: 4,
        };
        // Projections already completed before this private production stage;
        // only the real per-head normalization tensors are consumed here.
        let unused = Tensor {
            name: String::from("unused-prefill-projection"),
            dimensions: vec![256, 256],
            kind: 0,
            data: Arc::from([]),
        };
        let weights = FullAttention {
            query: unused.clone(),
            key: unused.clone(),
            value: unused.clone(),
            output: unused,
            query_norm: vec![1.0; geometry.head],
            key_norm: vec![1.0; geometry.head],
        };
        let base = 63;
        let mut keys = KvCache::new(geometry.kv_width(), 128)?;
        let mut values = KvCache::new(geometry.kv_width(), 128)?;
        for position in 0..base {
            let key: Vec<_> = (0..geometry.kv_width())
                .map(|index| sample(index + position))
                .collect();
            let value: Vec<_> = (0..geometry.kv_width())
                .map(|index| sample(index + 3 * position + 1))
                .collect();
            keys.reserve_append()?;
            values.reserve_append()?;
            keys.append(&key);
            values.append(&value);
        }
        let parallelism = RowParallelism::new(workers)?;
        let mut tile = TiledPrefill::new(
            count,
            geometry.query_gate_width(),
            0,
            MatvecBackend::Scalar,
            parallelism.clone(),
        )?;
        for token in 0..count {
            tile.mixed[token] = (0..geometry.query_gate_width())
                .map(|index| sample(index + 2 * token + 2))
                .collect();
            tile.recurrent_gate[token] = (0..geometry.kv_width())
                .map(|index| sample(index + 3 * token + 3))
                .collect();
            tile.beta[token] = (0..geometry.kv_width())
                .map(|index| sample(index + 4 * token + 4))
                .collect();
        }
        let mut projection = super::super::ProjectionScratch::default();
        projection.matvec.set_row_parallelism(parallelism);
        Ok(Self {
            geometry,
            weights,
            keys,
            values,
            projection,
            transforms: Transforms::default(),
            adapter_scratch: AdapterScratch::default(),
            attention_scratch: super::super::AttentionScratch::default(),
            tile,
            base,
            count,
        })
    }

    fn run(&mut self, arithmetic: Arithmetic, check: &dyn Fn() -> Result<()>) -> Result<()> {
        let mut frequencies_ready = false;
        full_attention_tile(
            &self.weights,
            &self.geometry,
            arithmetic,
            &self.transforms,
            self.base,
            self.count,
            &mut frequencies_ready,
            &mut self.keys,
            &mut self.values,
            &mut self.projection,
            None,
            0.0,
            &mut self.adapter_scratch,
            &mut self.attention_scratch,
            &mut self.tile,
            check,
        )
    }

    fn assert_same(&self, expected: &Self) {
        assert!(
            self.tile
                .attended
                .iter()
                .flatten()
                .map(|value| value.to_bits())
                .eq(expected
                    .tile
                    .attended
                    .iter()
                    .flatten()
                    .map(|value| value.to_bits()))
        );
        assert!(
            self.keys
                .tokens()
                .flatten()
                .eq(expected.keys.tokens().flatten())
        );
        assert!(
            self.values
                .tokens()
                .flatten()
                .eq(expected.values.tokens().flatten())
        );
    }
}

#[test]
fn six_head_production_prefill_preserves_causal_state_dispatch_and_cancellation() -> Result<()> {
    #[cfg(feature = "parallel-rows")]
    let worker_counts = [2, 4, 18];
    #[cfg(not(feature = "parallel-rows"))]
    let worker_counts = [1];
    for count in [8, 16, 32] {
        for arithmetic in [Arithmetic::Source, Arithmetic::PinnedSwift] {
            let mut expected = Harness::new(count, 1)?;
            expected.run(arithmetic, &|| Ok(()))?;
            for workers in worker_counts {
                let mut actual = Harness::new(count, workers)?;
                actual.run(arithmetic, &|| Ok(()))?;
                actual.assert_same(&expected);
                if matches!(arithmetic, Arithmetic::PinnedSwift) && workers > 1 {
                    assert_eq!(
                        actual.attention_scratch.scores.len(),
                        count * 6 * (63 + count)
                    );
                }
            }
        }
    }
    let mut actual = Harness::new(8, worker_counts[0])?;
    let saved_keys = actual.keys.copy()?;
    let saved_values = actual.values.copy()?;
    let original_keys = actual.tile.recurrent_gate.clone();
    let original_values = actual.tile.beta.clone();
    let caller = std::thread::current().id();
    let calls = Cell::new(0);
    // Refuse publication immediately after a provisional cache append, before
    // the six-head dispatch. The owner restores both complete cache components.
    assert!(
        actual
            .run(Arithmetic::PinnedSwift, &|| {
                assert_eq!(std::thread::current().id(), caller);
                calls.set(calls.get() + 1);
                if calls.get() == 2 {
                    Err(invalid("cancelled six-head production prefill"))
                } else {
                    Ok(())
                }
            })
            .is_err()
    );
    assert_eq!(calls.get(), 2);
    actual.keys = saved_keys;
    actual.values = saved_values;
    actual.tile.recurrent_gate = original_keys;
    actual.tile.beta = original_values;
    actual.run(Arithmetic::PinnedSwift, &|| Ok(()))?;
    let mut expected = Harness::new(8, 1)?;
    expected.run(Arithmetic::PinnedSwift, &|| Ok(()))?;
    actual.assert_same(&expected);
    Ok(())
}

#[cfg(feature = "parallel-rows")]
#[test]
fn x1_tiled_qkv_query_failure_precedes_all_key_value_chunks() -> Result<()> {
    for arithmetic in [Arithmetic::Source, Arithmetic::PinnedSwift] {
        for workers in [2, 3, 4, 18] {
            // Thirteen tokens exercise unequal and one-token worker tails.
            let mut actual = Harness::new(13, workers)?;
            for projected in &mut actual.tile.mixed {
                projected.fill(0.0);
            }
            actual.tile.mixed[12][23 * 2 * actual.geometry.head] = 1.0;
            actual.weights.query_norm.fill(100_000.0);
            actual.tile.recurrent_gate[0][0] = f32::NAN;
            actual.tile.beta[0][0] = 70_000.0;
            assert!(actual.run(arithmetic, &|| Ok(())).is_err_and(|error| {
                error.to_string() == "nonfinite values in binary16 attention query"
            }));
            assert_eq!(actual.keys.tokens().count(), actual.base);
            assert_eq!(actual.values.tokens().count(), actual.base);
            assert!(actual.attention_scratch.scores.is_empty());
        }
    }
    Ok(())
}

#[cfg(feature = "parallel-rows")]
#[test]
fn x1_tiled_qkv_keeps_earliest_key_value_chunk_and_recovers() -> Result<()> {
    for arithmetic in [Arithmetic::Source, Arithmetic::PinnedSwift] {
        for workers in [2, 3, 4, 18] {
            let mut actual = Harness::new(13, workers)?;
            let original_keys = actual.tile.recurrent_gate.clone();
            let original_values = actual.tile.beta.clone();
            // The first task's value overflow outranks a later task's bad key,
            // preserving the original paired worker submission's error order.
            actual.tile.beta[0][0] = 70_000.0;
            actual.tile.recurrent_gate[12][0] = f32::NAN;
            assert!(actual.run(arithmetic, &|| Ok(())).is_err_and(|error| {
                error.to_string() == "nonfinite values in binary16 value cache"
            }));
            assert_eq!(actual.keys.tokens().count(), actual.base);
            assert_eq!(actual.values.tokens().count(), actual.base);
            assert!(actual.attention_scratch.scores.is_empty());
            // An owner restores provisional inputs before reusing the stage.
            actual.tile.recurrent_gate = original_keys;
            actual.tile.beta = original_values;
            actual.run(arithmetic, &|| Ok(()))?;
            let mut expected = Harness::new(13, 1)?;
            expected.run(arithmetic, &|| Ok(()))?;
            actual.assert_same(&expected);
        }
    }
    Ok(())
}
