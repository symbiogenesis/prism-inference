//! Reset-only terminal state must match ordinary evaluated generation.

use super::*;
use std::{cell::Cell, rc::Rc, time::Duration};

#[derive(Clone, Debug, Default, PartialEq)]
struct State {
    tokens: Vec<u32>,
    recurrent: [f32; 4],
    convolution: [f32; 3],
    kv: Vec<u16>,
}

struct Fixture {
    state: State,
    evaluations: usize,
    ingestions: usize,
    fail_ingest: Rc<Cell<bool>>,
    cancel_ingest: Option<Cancellation>,
}

impl Fixture {
    fn advance(&mut self, token: u32) -> Result<()> {
        let value = f32::from(u16::try_from(token)?);
        for (index, lane) in self.state.recurrent.iter_mut().enumerate() {
            *lane = lane.mul_add(0.5, value + f32::from(u16::try_from(index)?));
        }
        self.state.convolution.rotate_left(1);
        self.state.convolution[2] = value;
        self.state.kv.push(u16::try_from(token)?);
        self.state.tokens.push(token);
        Ok(())
    }
}

impl Decoder for Fixture {
    type Checkpoint = State;

    fn identity(&self) -> &'static str {
        "terminal-state-fixture/v1"
    }
    fn vocab_size(&self) -> usize {
        8
    }
    fn checkpoint_bytes(&self) -> usize {
        self.state.tokens.len() * 6 + 28
    }
    fn reset(&mut self) -> Result<()> {
        self.state = State::default();
        Ok(())
    }
    fn evaluate(&mut self, token: u32) -> Result<Vec<f32>> {
        self.evaluations += 1;
        self.advance(token)?;
        let selected = usize::try_from(self.state.tokens.iter().sum::<u32>() % 8)?;
        let mut logits = vec![-1.0; 8];
        logits[selected] = 1.0;
        Ok(logits)
    }
    fn ingest(&mut self, token: u32) -> Result<()> {
        self.ingestions += 1;
        self.advance(token)?;
        if let Some(cancellation) = &self.cancel_ingest {
            cancellation.cancel();
        }
        if self.fail_ingest.replace(false) {
            return Err(std::io::Error::other("injected terminal state failure").into());
        }
        Ok(())
    }
    fn checkpoint(&self) -> Result<State> {
        Ok(self.state.clone())
    }
    fn restore(&mut self, checkpoint: &State) -> Result<()> {
        self.state.clone_from(checkpoint);
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct Seeded {
    state: u64,
    draws: usize,
}

impl Default for Seeded {
    fn default() -> Self {
        Self {
            state: 42,
            draws: 0,
        }
    }
}

impl Sampler for Seeded {
    fn sample(&mut self, logits: &[f32]) -> Result<u32> {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.draws += 1;
        Ok((greedy(logits)? + u32::try_from(self.state % 8)?) % 8)
    }
}

fn runtime() -> Result<Runtime<Fixture>> {
    Runtime::new(
        Fixture {
            state: State::default(),
            evaluations: 0,
            ingestions: 0,
            fail_ingest: Rc::new(Cell::new(false)),
            cancel_ingest: None,
        },
        Limits {
            context_tokens: 64,
            checkpoints: 0,
            checkpoint_bytes: 16_384,
        },
    )
}

fn deadline() -> Instant {
    Instant::now() + Duration::from_secs(60)
}

#[test]
fn terminal_state_only_matches_tokens_state_rng_and_counts_then_recomputes() -> Result<()> {
    for max_tokens in [1, 4, 8] {
        for timed in [false, true] {
            let (mut ordinary, mut reset_only) = (runtime()?, runtime()?);
            let (mut reference, mut sampler) = (Seeded::default(), Seeded::default());
            let options = Generation {
                max_tokens,
                stop_tokens: vec![],
            };
            let (mut expected_output, mut output) = (Vec::new(), Vec::new());
            let expected = ordinary.generate_sampled_until(
                &[1, 2, 3],
                &options,
                Prefill::default(),
                &Cancellation::default(),
                deadline(),
                &mut reference,
                |token| {
                    expected_output.push(token);
                    Ok(())
                },
            )?;
            let mut timings = GenerationTimings::default();
            let actual = reset_only.generate_sampled_until_for_reset(
                &[1, 2, 3],
                &options,
                Prefill::default(),
                &Cancellation::default(),
                deadline(),
                &mut sampler,
                timed.then_some(&mut timings),
                |token| {
                    output.push(token);
                    Ok(())
                },
            )?;
            assert_eq!(actual, expected);
            assert_eq!(output, expected_output);
            assert_eq!(sampler, reference);
            assert_eq!(reset_only.tokens(), ordinary.tokens());
            assert_eq!(reset_only.decoder().state, ordinary.decoder().state);
            assert_eq!(
                reset_only.decoder().evaluations + 1,
                ordinary.decoder().evaluations
            );
            assert_eq!(
                reset_only.decoder().ingestions,
                ordinary.decoder().ingestions + 1
            );
            assert!(reset_only.logits.is_empty());
            assert_eq!(reset_only.checkpoint_count(), 0);
            if timed {
                assert!(timings.succeeded);
                assert_eq!(
                    timings
                        .phase(GenerationPhase::TerminalIngestion)
                        .completed_tokens_lower_bound,
                    1
                );
                assert_eq!(
                    timings
                        .phase(GenerationPhase::DecodeEvaluation)
                        .completed_tokens_lower_bound,
                    max_tokens - 1
                );
            }

            // An accidental ordinary continuation must recompute the complete
            // prefix, rather than sampling logits from the penultimate token.
            let prompt = reset_only.tokens().to_vec();
            let mut fresh = runtime()?;
            let completion = reset_only.generate_sampled_until(
                &prompt,
                &options,
                Prefill::default(),
                &Cancellation::default(),
                deadline(),
                &mut sampler,
                |_| Ok(()),
            )?;
            fresh.generate_sampled_until(
                &prompt,
                &options,
                Prefill::default(),
                &Cancellation::default(),
                deadline(),
                &mut reference,
                |_| Ok(()),
            )?;
            assert_eq!(completion.reused_tokens, 0);
            assert_eq!(reset_only.decoder().state, fresh.decoder().state);
            assert_eq!(reset_only.logits, fresh.logits);
            assert_eq!(sampler, reference);
        }
    }
    Ok(())
}

#[test]
fn terminal_state_only_stop_draw_has_no_ingestion_or_output() -> Result<()> {
    let mut runtime = runtime()?;
    let mut sampler = Seeded::default();
    let mut reference = sampler.clone();
    let mut logits = vec![-1.0; 8];
    logits[1] = 1.0;
    let stop = reference.sample(&logits)?;
    let mut timings = GenerationTimings::default();
    let completion = runtime.generate_sampled_until_for_reset(
        &[1],
        &Generation {
            max_tokens: 1,
            stop_tokens: vec![stop],
        },
        Prefill::default(),
        &Cancellation::default(),
        deadline(),
        &mut sampler,
        Some(&mut timings),
        |_| Err(std::io::Error::other("stop token must not emit").into()),
    )?;
    assert_eq!(completion.finish, Finish::Stop);
    assert_eq!(completion.generated_tokens, 0);
    assert_eq!(runtime.tokens(), [1]);
    assert_eq!(runtime.decoder().ingestions, 0);
    assert_eq!(sampler, reference);
    assert_eq!(
        timings
            .phase(GenerationPhase::TerminalIngestion)
            .attempted_calls,
        0
    );
    assert!(!runtime.logits.is_empty());
    Ok(())
}

#[test]
fn terminal_state_only_failure_cancellation_and_delivery_restore_state_and_sampler() -> Result<()> {
    for fault in 0..3 {
        let mut runtime = runtime()?;
        let options = Generation {
            max_tokens: 1,
            stop_tokens: vec![],
        };
        runtime.generate(&[1], &options, &Cancellation::default(), |_| Ok(()))?;
        let old_state = runtime.decoder().state.clone();
        let old_tokens = runtime.tokens().to_vec();
        let old_logits = runtime.logits.clone();
        let mut sampler = Seeded::default();
        let old_sampler = sampler.clone();
        let cancellation = Cancellation::default();
        if fault == 0 {
            runtime.decoder.fail_ingest.set(true);
        } else if fault == 1 {
            runtime.decoder.cancel_ingest = Some(cancellation.clone());
        }
        let mut timings = GenerationTimings::default();
        let mut calls = 0;
        let result = runtime.generate_sampled_until_for_reset(
            &old_tokens,
            &options,
            Prefill::default(),
            &cancellation,
            deadline(),
            &mut sampler,
            Some(&mut timings),
            |_| {
                calls += 1;
                Err(std::io::Error::other("injected terminal delivery failure").into())
            },
        );
        assert!(result.is_err());
        assert_eq!(calls, usize::from(fault == 2));
        assert_eq!(runtime.tokens(), old_tokens);
        assert_eq!(runtime.logits, old_logits);
        assert_eq!(runtime.decoder().state, old_state);
        assert_eq!(sampler, old_sampler);
        assert!(!runtime.is_faulted());
        assert_eq!(
            timings
                .phase(GenerationPhase::RollbackRestore)
                .completed_calls,
            1
        );
        assert_eq!(
            timings
                .phase(GenerationPhase::SamplerRestore)
                .completed_calls,
            1
        );
    }
    Ok(())
}

#[test]
fn terminal_state_only_rejects_retained_snapshots_and_expired_requests_before_work() -> Result<()> {
    for cached in [false, true] {
        let mut runtime = runtime()?;
        runtime.limits.checkpoints = usize::from(cached);
        let mut sampler = Seeded::default();
        let original = sampler.clone();
        let result = runtime.generate_sampled_until_for_reset(
            &[1],
            &Generation {
                max_tokens: 1,
                stop_tokens: vec![],
            },
            Prefill::default(),
            &Cancellation::default(),
            Instant::now(),
            &mut sampler,
            None,
            |_| Ok(()),
        );
        assert!(result.is_err());
        assert!(runtime.tokens().is_empty());
        assert_eq!(runtime.decoder().evaluations, 0);
        assert_eq!(runtime.decoder().ingestions, 0);
        assert_eq!(sampler, original);
    }
    Ok(())
}
