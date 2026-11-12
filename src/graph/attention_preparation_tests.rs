//! Decode preparation parity and failure ordering across aligned worker groups.

use std::sync::Arc;

use super::{
    Arithmetic, AttentionScratch, Evaluation, FullAttention, Geometry, KvCache, ProjectionScratch,
    Transforms,
};
use crate::{Result, kernels::RowParallelism, model::Tensor};

struct Fixture {
    geometry: Geometry,
    layer: FullAttention,
    projected: Vec<f32>,
    key: Vec<f32>,
    value: Vec<f32>,
    keys: KvCache,
    values: KvCache,
    scratch: AttentionScratch,
}

impl Fixture {
    fn new() -> Result<Self> {
        let geometry = Geometry {
            embedding: 256,
            feed_forward: 256,
            heads: 24,
            kv_heads: 4,
            head: 256,
            rope: 64,
            rope_sections: [11, 11, 10, 0],
            rope_base: 10_000_000.0,
            epsilon: 0.000_001,
            ssm_head: 128,
            ssm_keys: 2,
            ssm_values: 4,
            convolution: 4,
        };
        // Projection tensors are not consumed by this preparation/state control.
        let unused = Tensor {
            name: "unused-preparation-control".into(),
            dimensions: vec![256, 256],
            kind: 0,
            data: Arc::from([]),
        };
        let layer = FullAttention {
            query: unused.clone(),
            key: unused.clone(),
            value: unused.clone(),
            output: unused,
            query_norm: vec![1.0; geometry.head],
            key_norm: vec![1.0; geometry.head],
        };
        let pattern = [-0.75, -0.0, 0.25, 0.5, -0.125, 1.0, -0.5];
        let patterned = |len| {
            (0..len)
                .map(|index| pattern[index % pattern.len()])
                .collect()
        };
        let kv_width = geometry.kv_heads * geometry.head;
        let mut keys = KvCache::new(kv_width, 3)?;
        let mut values = KvCache::new(kv_width, 3)?;
        for token in 0..2 {
            let past: Vec<f32> = (0..kv_width)
                .map(|index| pattern[(index + token) % pattern.len()])
                .collect();
            keys.reserve_append()?;
            values.reserve_append()?;
            keys.append(&past);
            values.append(&past);
        }
        Ok(Self {
            projected: patterned(geometry.query_gate_width()),
            key: patterned(kv_width),
            value: patterned(kv_width),
            geometry,
            layer,
            keys,
            values,
            scratch: AttentionScratch::default(),
        })
    }

    fn run(&mut self, arithmetic: Arithmetic, workers: usize) -> Result<()> {
        let mut projection = ProjectionScratch::default();
        projection
            .matvec
            .set_row_parallelism(RowParallelism::new(workers)?);
        let transforms = Transforms::default();
        let mut adapter = crate::model::lora::Scratch::default();
        let mut evaluation = Evaluation {
            geometry: &self.geometry,
            arithmetic,
            transforms: &transforms,
            position: 2,
            frequencies_ready: false,
            projection: &mut projection,
            adapter: None,
            adapter_scale: 0.0,
            adapter_scratch: &mut adapter,
            executor: None,
        };
        evaluation.full_attention_state(
            &self.layer,
            &self.projected,
            &mut self.key,
            &mut self.value,
            &self.keys,
            &self.values,
            &mut self.scratch.query,
            &mut self.scratch.scores,
            &mut self.scratch.frequencies,
            &mut self.scratch.attended,
        )
    }

    fn assert_unpublished(&self) {
        assert_eq!(self.scratch.attended.len(), 0);
        assert_eq!(self.scratch.scores.len(), 0);
        assert_eq!(self.keys.tokens().count(), 2);
        assert_eq!(self.values.tokens().count(), 2);
    }
}

fn assert_bits_equal(expected: &[f32], actual: &[f32]) {
    assert!(
        expected
            .iter()
            .map(|v| v.to_bits())
            .eq(actual.iter().map(|v| v.to_bits()))
    );
}

#[test]
fn attention_preparation_preserves_serial_bits_across_worker_partitions() -> Result<()> {
    #[cfg(feature = "parallel-rows")]
    let worker_counts = [1, 2, 3, 4, 7, 18];
    #[cfg(not(feature = "parallel-rows"))]
    let worker_counts = [1];
    for arithmetic in [Arithmetic::Source, Arithmetic::PinnedSwift] {
        let mut expected = Fixture::new()?;
        expected.run(arithmetic, 1)?;
        for workers in worker_counts {
            let mut actual = Fixture::new()?;
            actual.run(arithmetic, workers)?;
            assert_bits_equal(&expected.scratch.query, &actual.scratch.query);
            assert_bits_equal(&expected.key, &actual.key);
            assert_bits_equal(&expected.value, &actual.value);
            assert_bits_equal(&expected.scratch.attended, &actual.scratch.attended);
        }
    }
    Ok(())
}

#[test]
fn attention_preparation_query_failure_precedes_earlier_key_value_failures() -> Result<()> {
    #[cfg(feature = "parallel-rows")]
    let worker_counts = [1, 2, 3, 4, 7, 18];
    #[cfg(not(feature = "parallel-rows"))]
    let worker_counts = [1];
    for arithmetic in [Arithmetic::Source, Arithmetic::PinnedSwift] {
        for workers in worker_counts {
            let mut fixture = Fixture::new()?;
            fixture.projected.fill(0.0);
            fixture.projected[23 * 2 * fixture.geometry.head] = 1.0;
            fixture.layer.query_norm.fill(100_000.0);
            fixture.key[0] = f32::NAN;
            fixture.value[0] = 70_000.0;
            assert!(fixture.run(arithmetic, workers).is_err_and(|error| {
                error.to_string() == "nonfinite values in binary16 attention query"
            }));
            fixture.assert_unpublished();
        }
    }
    Ok(())
}

#[cfg(feature = "parallel-rows")]
#[test]
fn attention_preparation_keeps_earliest_key_value_chunk_failure() -> Result<()> {
    for arithmetic in [Arithmetic::Source, Arithmetic::PinnedSwift] {
        for workers in [2, 4] {
            let mut fixture = Fixture::new()?;
            // The first task's value overflow must outrank a later task's
            // malformed key, matching the original paired-dispatch order.
            fixture.value[0] = 70_000.0;
            fixture.key[3 * fixture.geometry.head] = f32::NAN;
            assert!(fixture.run(arithmetic, workers).is_err_and(|error| {
                error.to_string() == "nonfinite values in binary16 value cache"
            }));
            fixture.assert_unpublished();
        }
    }
    Ok(())
}
