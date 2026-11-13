//! Sampled suffix serving against independent ordinary hybrid-graph execution.

use super::*;
use bonsai_prism_runtime::graph::KvCachePolicy;
use bonsai_prism_runtime::sampling::{FiniteLogits, ReferenceSampler, Sampler, Sampling};
use bonsai_prism_runtime::speculative::SuffixDraftConfig;
use std::time::{Duration, Instant};

#[derive(Clone)]
struct TracedSampler {
    sampler: ReferenceSampler,
    rows: Vec<Vec<u32>>,
}

impl TracedSampler {
    fn new(seed: u64) -> Result<Self> {
        Ok(Self {
            sampler: ReferenceSampler::new(
                Sampling {
                    temperature: 0.9,
                    top_k: 0,
                    top_p: 1.0,
                },
                seed,
            )?,
            rows: Vec::new(),
        })
    }
}

impl Sampler for TracedSampler {
    fn sample(&mut self, logits: &[f32]) -> Result<u32> {
        self.sample_finite(FiniteLogits::new(logits)?)
    }

    fn sample_finite(&mut self, logits: FiniteLogits<'_>) -> Result<u32> {
        self.rows.push(logits.as_slice().iter().map(|value| value.to_bits()).collect());
        self.sampler.sample_finite(logits)
    }
}

fn graph(policy: KvCachePolicy, tiled: bool) -> Result<Qwen35> {
    let mut model = super::quantized_kv::profile_fixture();
    model.metadata.insert("qwen35.context_length".into(), MetadataValue::from(96_u64));
    let graph = Qwen35::with_state_budget_and_kv(
        model, 96, 1_048_576, MatvecBackend::Auto, policy,
    )?;
    if !tiled {
        return Ok(graph);
    }
    let bytes = graph.tiled_prefill_scratch_bytes(8)?;
    let graph = graph.with_tiled_prefill(8, bytes)?;
    let bytes = graph.tiled_verification_scratch_bytes()?;
    graph.with_tiled_verification(bytes)
}

const LIMITS: Limits = Limits {
    context_tokens: 96,
    checkpoints: 0,
    checkpoint_bytes: 4 * 1_048_576,
};

fn deadline() -> Instant {
    Instant::now() + Duration::from_secs(30)
}

#[test]
fn sampled_suffix_matches_target_draws_full_state_and_reused_continuation() -> Result<()> {
    for policy in [KvCachePolicy::F16, KvCachePolicy::Rotated4, KvCachePolicy::Rotated4Keys3Values] {
        for seed in [17, 4242] {
            let mut actual = Runtime::new(graph(policy, true)?, LIMITS)?;
            let mut expected = Runtime::new(graph(policy, false)?, LIMITS)?;
            let mut actual_sampler = TracedSampler::new(seed)?;
            let mut expected_sampler = TracedSampler::new(seed)?;
            let mut prompt: Vec<_> = [0, 1, 0, 1].into_iter().cycle().take(63).collect();
            for _ in 0..2 {
                let options = Generation { max_tokens: 9, stop_tokens: Vec::new() };
                let mut actual_output = Vec::new();
                let mut expected_output = Vec::new();
                let cancellation = Cancellation::default();
                let (completion, stats) = actual.generate_sampled_speculative_until(
                    &prompt, &options, Prefill { block_tokens: 8 },
                    SuffixDraftConfig::new(1, 64, 8)?, &cancellation, deadline(),
                    &mut actual_sampler, |token| { actual_output.push(token); Ok(()) },
                )?;
                let reference = expected.generate_sampled_until(
                    &prompt, &options, Prefill { block_tokens: 8 },
                    &cancellation, deadline(), &mut expected_sampler,
                    |token| { expected_output.push(token); Ok(()) },
                )?;
                assert!(stats.draft_rounds > 0);
                assert_eq!(completion, reference);
                assert_eq!(actual_output, expected_output);
                assert_eq!(actual_sampler.rows, expected_sampler.rows);
                assert_eq!(actual.tokens(), expected.tokens());
                assert_eq!(graph_bits(actual.decoder())?, graph_bits(expected.decoder())?);
                assert!(actual.decoder().tiled_prefill_statistics().completed_tiles > 0);
                prompt = actual.tokens().to_vec();
            }
        }
    }
    Ok(())
}

#[derive(Clone)]
struct StopAt {
    position: usize,
    stop: usize,
}

impl Sampler for StopAt {
    fn sample(&mut self, logits: &[f32]) -> Result<u32> {
        FiniteLogits::new(logits)?;
        let token = u32::from(self.position == self.stop);
        self.position += 1;
        Ok(token)
    }
}

#[test]
fn sampled_suffix_stops_inside_a_quantized_tile_without_committing_the_suffix() -> Result<()> {
    for policy in [KvCachePolicy::F16, KvCachePolicy::Rotated4, KvCachePolicy::Rotated4Keys3Values] {
        for stop in [0, 1, 3, 7] {
            let mut actual = Runtime::new(graph(policy, true)?, LIMITS)?;
            let mut expected = Runtime::new(graph(policy, false)?, LIMITS)?;
            let prompt = [0; 63];
            let options = Generation { max_tokens: 9, stop_tokens: vec![1] };
            let cancellation = Cancellation::default();
            let mut actual_sampler = StopAt { position: 0, stop };
            let mut expected_sampler = actual_sampler.clone();
            let (completion, _) = actual.generate_sampled_speculative_until(
                &prompt, &options, Prefill { block_tokens: 8 },
                SuffixDraftConfig::new(1, 64, 8)?, &cancellation, deadline(),
                &mut actual_sampler, |_| Ok(()),
            )?;
            let reference = expected.generate_sampled_until(
                &prompt, &options, Prefill { block_tokens: 8 },
                &cancellation, deadline(), &mut expected_sampler, |_| Ok(()),
            )?;
            assert_eq!(completion, reference);
            assert_eq!(completion.finish, bonsai_prism_runtime::runtime::Finish::Stop);
            assert_eq!(completion.generated_tokens, stop);
            assert_eq!(actual_sampler.position, expected_sampler.position);
            assert_eq!(actual.tokens(), expected.tokens());
            assert_eq!(graph_bits(actual.decoder())?, graph_bits(expected.decoder())?);
        }
    }
    Ok(())
}

#[test]
fn sampled_suffix_delivery_failure_restores_graph_prefix_and_sampler_draws() -> Result<()> {
    for policy in [KvCachePolicy::F16, KvCachePolicy::Rotated4, KvCachePolicy::Rotated4Keys3Values] {
        let mut actual = Runtime::new(graph(policy, true)?, LIMITS)?;
        let cancellation = Cancellation::default();
        let mut sampler = TracedSampler::new(4242)?;
        actual.generate_sampled_until(
            &[0, 1], &Generation { max_tokens: 1, stop_tokens: Vec::new() },
            Prefill::default(), &cancellation, deadline(), &mut sampler, |_| Ok(()),
        )?;
        let tokens = actual.tokens().to_vec();
        let state = graph_bits(actual.decoder())?;
        let mut original_sampler = sampler.clone();
        let prompt: Vec<_> = [0, 1, 0, 1].into_iter().cycle().take(63).collect();
        let mut delivered = 0;
        assert!(actual.generate_sampled_speculative_until(
            &prompt, &Generation { max_tokens: 9, stop_tokens: Vec::new() },
            Prefill { block_tokens: 8 }, SuffixDraftConfig::new(1, 64, 8)?,
            &cancellation, deadline(), &mut sampler, |_| {
                delivered += 1;
                if delivered == 2 { Err("injected sampled graph delivery failure".into()) } else { Ok(()) }
            },
        ).is_err());
        assert_eq!(delivered, 2);
        assert_eq!(actual.tokens(), tokens);
        assert_eq!(graph_bits(actual.decoder())?, state);
        assert!(!actual.is_faulted());
        assert_eq!(sampler.rows, original_sampler.rows);
        assert_eq!(sampler.sample(&[0.6, -0.4])?, original_sampler.sample(&[0.6, -0.4])?);
    }
    Ok(())
}
