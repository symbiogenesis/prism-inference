#![forbid(unsafe_code)]
//! Recurrent-state ownership regressions independent of a model artifact.

#[path = "runtime/timing.rs"]
mod timing;

use bonsai_prism_runtime::{
    Decoder, MAX_PREFILL_BLOCK_TOKENS, Result,
    runtime::{Cancellation, Finish, Generation, Limits, Prefill, Runtime},
    sampling::{FiniteLogits, Sampler},
};
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
    time::{Duration, Instant},
};

struct Recurrent {
    tokens: Vec<u32>,
    steps: Rc<Cell<usize>>,
    fail: Rc<Cell<bool>>,
    restore_fail: Rc<Cell<bool>>,
    checkpoint_fail: Rc<Cell<bool>>,
    reset_fail: Rc<Cell<bool>>,
}

impl Decoder for Recurrent {
    type Checkpoint = Vec<u32>;

    fn identity(&self) -> &'static str {
        "recurrent-fixture/f32/fp16-kv/v1"
    }
    fn vocab_size(&self) -> usize {
        8
    }
    fn checkpoint_bytes(&self) -> usize {
        self.tokens.len() * 4
    }
    fn reset(&mut self) -> Result<()> {
        self.tokens.clear();
        if self.reset_fail.get() {
            return Err(
                std::io::Error::other("injected reset failure after state mutation").into(),
            );
        }
        Ok(())
    }
    fn evaluate(&mut self, token: u32) -> Result<Vec<f32>> {
        self.steps.set(self.steps.get() + 1);
        self.tokens.push(token);
        if self.fail.replace(false) {
            return Err(std::io::Error::other("injected failure after state mutation").into());
        }
        let next = usize::try_from(self.tokens.iter().sum::<u32>() % 8)?;
        let mut logits = vec![-1.0; 8];
        logits[next] = 1.0;
        Ok(logits)
    }
    fn checkpoint(&self) -> Result<Self::Checkpoint> {
        if self.checkpoint_fail.replace(false) {
            return Err(std::io::Error::other("injected checkpoint allocation failure").into());
        }
        Ok(self.tokens.clone())
    }
    fn restore(&mut self, checkpoint: &Self::Checkpoint) -> Result<()> {
        if self.restore_fail.get() {
            return Err(std::io::Error::other("injected restore failure").into());
        }
        self.tokens.clone_from(checkpoint);
        Ok(())
    }
}

fn decoder() -> Recurrent {
    Recurrent {
        tokens: Vec::new(),
        steps: Rc::new(Cell::new(0)),
        fail: Rc::new(Cell::new(false)),
        restore_fail: Rc::new(Cell::new(false)),
        checkpoint_fail: Rc::new(Cell::new(false)),
        reset_fail: Rc::new(Cell::new(false)),
    }
}

const LIMITS: Limits = Limits {
    context_tokens: 32,
    checkpoints: 4,
    checkpoint_bytes: 16_384,
};

const fn options() -> Generation {
    Generation {
        max_tokens: 1,
        stop_tokens: vec![],
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
enum SamplingFault {
    #[default]
    None,
    Error,
    InvalidToken,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct SequenceSampler {
    state: u64,
    draws: usize,
    fault: SamplingFault,
}

impl Default for SequenceSampler {
    fn default() -> Self {
        Self {
            state: 42,
            draws: 0,
            fault: SamplingFault::None,
        }
    }
}

impl Sampler for SequenceSampler {
    fn sample(&mut self, logits: &[f32]) -> Result<u32> {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.draws += 1;
        match self.fault {
            SamplingFault::None => Ok(u32::try_from(self.state % u64::try_from(logits.len())?)?),
            SamplingFault::Error => {
                Err(std::io::Error::other("sampler failed after RNG advancement").into())
            }
            SamplingFault::InvalidToken => Ok(u32::try_from(logits.len())?),
        }
    }
}

#[test]
fn sampled_success_commits_caller_rng_and_stop_draws_without_emitting_stops() -> Result<()> {
    let mut runtime = Runtime::new(scheduled(), LIMITS)?;
    let mut sampler = SequenceSampler::default();
    let mut expected_sampler = sampler.clone();
    let expected = [
        expected_sampler.sample(&[0.0; 8])?,
        expected_sampler.sample(&[0.0; 8])?,
    ];
    let stop = expected_sampler.sample(&[0.0; 8])?;
    let mut actual = Vec::new();
    let completion = runtime.generate_sampled_until(
        &[1, 2, 3, 4],
        &Generation {
            max_tokens: 4,
            stop_tokens: vec![stop],
        },
        Prefill { block_tokens: 2 },
        &Cancellation::default(),
        Instant::now() + Duration::from_mins(1),
        &mut sampler,
        |token| {
            actual.push(token);
            Ok(())
        },
    )?;
    assert_eq!(actual, expected);
    assert_eq!(sampler, expected_sampler);
    assert_eq!(completion.finish, Finish::Stop);
    assert_eq!(completion.generated_tokens, 2);
    assert_eq!(runtime.tokens(), &[1, 2, 3, 4, expected[0], expected[1]]);
    assert_eq!(runtime.decoder().blocks, [vec![1, 2], vec![3]]);
    Ok(())
}

#[test]
fn sampled_callback_failure_restores_rng_and_retry_repeats_output() -> Result<()> {
    let mut runtime = Runtime::new(decoder(), LIMITS)?;
    let cancellation = Cancellation::default();
    runtime.generate(&[1], &options(), &cancellation, |_| Ok(()))?;
    let committed = runtime.tokens().to_vec();
    let generation = Generation {
        max_tokens: 3,
        stop_tokens: vec![],
    };
    let mut sampler = SequenceSampler::default();
    let original = sampler.clone();
    let mut failed_output = Vec::new();
    assert!(
        runtime
            .generate_sampled(
                &[2, 3],
                &generation,
                Prefill::default(),
                &cancellation,
                &mut sampler,
                |token| {
                    failed_output.push(token);
                    if failed_output.len() == 2 {
                        return Err(std::io::Error::other("consumer disconnected").into());
                    }
                    Ok(())
                },
            )
            .is_err()
    );
    assert_eq!(sampler, original);
    assert_eq!(runtime.tokens(), committed);
    assert_eq!(runtime.decoder().tokens, committed);
    let mut retried = Vec::new();
    runtime.generate_sampled(
        &[2, 3],
        &generation,
        Prefill::default(),
        &cancellation,
        &mut sampler,
        |token| {
            retried.push(token);
            Ok(())
        },
    )?;
    assert_eq!(&retried[..failed_output.len()], failed_output);
    assert_eq!(sampler.draws, 3);
    Ok(())
}

#[test]
fn sampled_errors_and_invalid_tokens_restore_rng_and_decoder_before_retry() -> Result<()> {
    for fault in [SamplingFault::Error, SamplingFault::InvalidToken] {
        let mut runtime = Runtime::new(decoder(), LIMITS)?;
        let cancellation = Cancellation::default();
        runtime.generate(&[1], &options(), &cancellation, |_| Ok(()))?;
        let mut sampler = SequenceSampler {
            fault,
            ..SequenceSampler::default()
        };
        let original = sampler.clone();
        let steps = runtime.decoder().steps.get();
        let mut emitted = Vec::new();
        assert!(
            runtime
                .generate_sampled(
                    &[2, 3],
                    &options(),
                    Prefill::default(),
                    &cancellation,
                    &mut sampler,
                    |token| {
                        emitted.push(token);
                        Ok(())
                    },
                )
                .is_err()
        );
        assert_eq!(sampler, original);
        assert_eq!(runtime.tokens(), &[1, 1]);
        assert_eq!(runtime.decoder().tokens, [1, 1]);
        assert_eq!(runtime.decoder().steps.get(), steps + 2);
        assert_eq!(emitted, [0_u32; 0]);
        assert!(!runtime.is_faulted());
        sampler.fault = SamplingFault::None;
        let expected = sampler.clone().sample(&[0.0; 8])?;
        runtime.generate_sampled(
            &[2, 3],
            &options(),
            Prefill::default(),
            &cancellation,
            &mut sampler,
            |token| {
                assert_eq!(token, expected);
                Ok(())
            },
        )?;
        assert_eq!(sampler.draws, 1);
    }
    Ok(())
}

#[derive(Clone)]
struct CancellingSampler {
    inner: SequenceSampler,
    cancellation: Cancellation,
}

impl Sampler for CancellingSampler {
    fn sample(&mut self, logits: &[f32]) -> Result<u32> {
        let token = self.inner.sample(logits)?;
        self.cancellation.cancel();
        Ok(token)
    }
}

#[test]
fn sampled_cancellation_after_draw_prevents_stop_or_decoder_commit_and_restores_rng() -> Result<()>
{
    for stop_tokens in [vec![], vec![1]] {
        let mut runtime = Runtime::new(decoder(), LIMITS)?;
        runtime.generate(&[1], &options(), &Cancellation::default(), |_| Ok(()))?;
        let cancellation = Cancellation::default();
        let mut sampler = CancellingSampler {
            inner: SequenceSampler::default(),
            cancellation: cancellation.clone(),
        };
        let original = sampler.inner.clone();
        let steps = runtime.decoder().steps.get();
        assert!(
            runtime
                .generate_sampled(
                    &[1, 1],
                    &Generation {
                        max_tokens: 2,
                        stop_tokens,
                    },
                    Prefill::default(),
                    &cancellation,
                    &mut sampler,
                    |_| Err(std::io::Error::other("cancelled sample must not emit").into()),
                )
                .is_err()
        );
        assert_eq!(sampler.inner, original);
        assert_eq!(runtime.tokens(), &[1, 1]);
        assert_eq!(runtime.decoder().tokens, [1, 1]);
        assert_eq!(runtime.decoder().steps.get(), steps);
        assert!(!runtime.is_faulted());
    }
    Ok(())
}

#[test]
fn sampled_decoder_and_recovery_failures_restore_rng_even_when_admission_closes() -> Result<()> {
    for fail_restore in [false, true] {
        let decoder = decoder();
        let fail = decoder.fail.clone();
        let restore_fail = decoder.restore_fail.clone();
        let mut runtime = Runtime::new(decoder, LIMITS)?;
        let cancellation = Cancellation::default();
        runtime.generate(&[1], &options(), &cancellation, |_| Ok(()))?;
        let mut sampler = SequenceSampler::default();
        let original = sampler.clone();
        let mut emitted = Vec::new();
        assert!(
            runtime
                .generate_sampled(
                    &[2],
                    &Generation {
                        max_tokens: 3,
                        stop_tokens: vec![],
                    },
                    Prefill::default(),
                    &cancellation,
                    &mut sampler,
                    |token| {
                        emitted.push(token);
                        fail.set(true);
                        restore_fail.set(fail_restore);
                        Ok(())
                    },
                )
                .is_err()
        );
        assert_eq!(emitted.len(), 1);
        assert_eq!(sampler, original);
        assert_eq!(runtime.tokens(), &[1, 1]);
        assert_eq!(runtime.is_faulted(), fail_restore);
        restore_fail.set(false);
        if fail_restore {
            assert!(
                runtime
                    .generate_sampled(
                        &[2],
                        &options(),
                        Prefill::default(),
                        &cancellation,
                        &mut sampler,
                        |_| Ok(()),
                    )
                    .is_err()
            );
            assert_eq!(sampler, original);
            runtime.reset()?;
        } else {
            assert_eq!(runtime.decoder().tokens, [1, 1]);
        }
        runtime.generate_sampled(
            &[2],
            &options(),
            Prefill::default(),
            &cancellation,
            &mut sampler,
            |token| {
                assert_eq!(token, emitted[0]);
                Ok(())
            },
        )?;
    }
    Ok(())
}

#[test]
fn sampled_rejected_requests_leave_rng_and_committed_graph_unchanged() -> Result<()> {
    let decoder = decoder();
    let checkpoint_fail = decoder.checkpoint_fail.clone();
    let mut runtime = Runtime::new(decoder, LIMITS)?;
    let cancellation = Cancellation::default();
    runtime.generate(&[1], &options(), &cancellation, |_| Ok(()))?;
    let mut sampler = SequenceSampler::default();
    let original = sampler.clone();
    let steps = runtime.decoder().steps.get();
    for prefill in [Prefill { block_tokens: 0 }, Prefill::default()] {
        checkpoint_fail.set(true);
        assert!(
            runtime
                .generate_sampled(
                    &[2],
                    &options(),
                    prefill,
                    &cancellation,
                    &mut sampler,
                    |_| Ok(()),
                )
                .is_err()
        );
        assert_eq!(sampler, original);
        assert_eq!(runtime.tokens(), &[1, 1]);
        assert_eq!(runtime.decoder().tokens, [1, 1]);
        assert_eq!(runtime.decoder().steps.get(), steps);
    }
    checkpoint_fail.set(true);
    assert!(
        runtime
            .generate_sampled_until(
                &[2],
                &options(),
                Prefill::default(),
                &cancellation,
                Instant::now(),
                &mut sampler,
                |_| Ok(()),
            )
            .is_err()
    );
    assert_eq!(sampler, original);
    assert_eq!(runtime.tokens(), &[1, 1]);
    assert_eq!(runtime.decoder().tokens, [1, 1]);
    assert_eq!(runtime.decoder().steps.get(), steps);
    assert!(checkpoint_fail.get());
    Ok(())
}

#[test]
fn edited_turns_restore_complete_recurrent_prefix_and_failed_stream_rolls_back() -> Result<()> {
    let decoder = decoder();
    let steps = decoder.steps.clone();
    let fail = decoder.fail.clone();
    let mut runtime = Runtime::new(decoder, LIMITS)?;
    let cancel = Cancellation::default();
    let first = runtime.generate(&[1], &options(), &cancel, |_| Ok(()))?;
    assert_eq!(first.generated_tokens, 1);
    assert_eq!(runtime.tokens(), &[1, 1]);
    runtime.generate(&[1, 1, 2], &options(), &cancel, |_| Ok(()))?;
    assert_eq!(runtime.tokens(), &[1, 1, 2, 4]);
    assert_eq!(steps.get(), 4);
    let edited = runtime.generate(&[1, 1, 3], &options(), &cancel, |_| Ok(()))?;
    assert_eq!(edited.reused_tokens, 2);
    assert_eq!(runtime.tokens(), &[1, 1, 3, 5]);
    assert_eq!(steps.get(), 6);

    let before = runtime.tokens().to_vec();
    let cancelled = Cancellation::default();
    assert!(
        runtime
            .generate(&[7, 2], &options(), &cancelled, |_| {
                cancelled.cancel();
                Ok(())
            })
            .is_err()
    );
    assert_eq!(runtime.tokens(), before);
    assert!(!runtime.is_faulted());

    fail.set(true);
    assert!(
        runtime
            .generate(&[6], &options(), &cancel, |_| Ok(()))
            .is_err()
    );
    assert_eq!(runtime.tokens(), before);
    let restored = runtime.generate(&[1, 1, 3, 5, 1], &options(), &cancel, |_| Ok(()))?;
    assert_eq!(restored.reused_tokens, 4);
    assert_eq!(runtime.tokens(), &[1, 1, 3, 5, 1, 3]);
    Ok(())
}

#[test]
fn edited_prefix_restoration_reuses_conversation_token_storage() -> Result<()> {
    let mut runtime = Runtime::new(decoder(), LIMITS)?;
    let token_storage = runtime.tokens().as_ptr();
    let cancellation = Cancellation::default();
    for (prompt, reused) in [
        (&[1][..], 0),
        (&[1, 1, 2], 2),
        (&[1, 1, 3], 2),
        (&[1, 1, 2, 4, 1], 4),
    ] {
        let completion = runtime.generate(prompt, &options(), &cancellation, |_| Ok(()))?;
        assert_eq!(runtime.tokens().as_ptr(), token_storage);
        assert_eq!(&runtime.tokens()[..prompt.len()], prompt);
        assert_eq!(completion.reused_tokens, reused);
    }
    assert_eq!(runtime.tokens(), &[1, 1, 2, 4, 1, 1]);
    Ok(())
}

#[test]
fn restoration_failure_closes_admission_until_reset() -> Result<()> {
    let decoder = decoder();
    let restore_fail = decoder.restore_fail.clone();
    let mut runtime = Runtime::new(decoder, LIMITS)?;
    let cancel = Cancellation::default();
    runtime.generate(&[1], &options(), &cancel, |_| Ok(()))?;
    let committed = runtime.tokens().to_vec();
    restore_fail.set(true);
    assert!(
        runtime
            .generate(&[2], &options(), &cancel, |_| {
                Err(std::io::Error::other("disconnected consumer").into())
            })
            .is_err()
    );
    assert!(runtime.is_faulted());
    // A failed backend restoration closes admission, but the public token
    // history must still describe the last successful transaction.
    assert_eq!(runtime.tokens(), committed);
    assert!(
        runtime
            .generate(&[1], &options(), &cancel, |_| Ok(()))
            .is_err()
    );
    assert_eq!(runtime.tokens(), committed);
    runtime.reset()?;
    assert_eq!(runtime.tokens(), &[0_u32; 0]);
    restore_fail.set(false);
    runtime.generate(&[3], &options(), &cancel, |_| Ok(()))?;
    assert_eq!(runtime.tokens(), &[3, 3]);
    Ok(())
}

#[test]
fn failed_reset_preserves_committed_history_and_latches_admission_until_success() -> Result<()> {
    let decoder = decoder();
    let reset_fail = decoder.reset_fail.clone();
    let steps = decoder.steps.clone();
    let mut runtime = Runtime::new(decoder, LIMITS)?;
    let cancellation = Cancellation::default();
    runtime.generate(&[1], &options(), &cancellation, |_| Ok(()))?;
    let committed = runtime.tokens().to_vec();
    let checkpoints = runtime.checkpoint_count();
    let completed_steps = steps.get();
    assert_eq!(runtime.decoder().tokens, committed);

    reset_fail.set(true);
    for _ in 0..2 {
        assert!(runtime.reset().is_err());
        assert!(runtime.is_faulted());
        assert_eq!(runtime.tokens(), committed);
        assert_eq!(runtime.checkpoint_count(), checkpoints);
        // A decoder may mutate before returning an error. Inspection exposes
        // that provisional backend state without rewriting committed history.
        assert_eq!(runtime.decoder().tokens, [0_u32; 0]);
        assert!(
            runtime
                .generate(&[2], &options(), &cancellation, |_| Ok(()))
                .is_err()
        );
        assert_eq!(steps.get(), completed_steps);
    }

    reset_fail.set(false);
    runtime.reset()?;
    assert!(!runtime.is_faulted());
    assert_eq!(runtime.tokens(), &[0_u32; 0]);
    assert_eq!(runtime.checkpoint_count(), 0);
    assert_eq!(runtime.decoder().tokens, [0_u32; 0]);
    let completion = runtime.generate(&[3], &options(), &cancellation, |_| Ok(()))?;
    assert_eq!(completion.reused_tokens, 0);
    assert_eq!(runtime.tokens(), &[3, 3]);
    Ok(())
}

#[test]
fn bounds_and_stop_tokens_do_not_silently_change_decoding() -> Result<()> {
    let decoder = decoder();
    let steps = decoder.steps.clone();
    let mut runtime = Runtime::new(decoder, LIMITS)?;
    let cancel = Cancellation::default();
    assert!(
        runtime
            .generate(&[8], &options(), &cancel, |_| Ok(()))
            .is_err()
    );
    assert!(
        runtime
            .generate(&[1; 32], &options(), &cancel, |_| Ok(()))
            .is_err()
    );
    assert_eq!(steps.get(), 0);
    let stop = Generation {
        max_tokens: 4,
        stop_tokens: vec![1],
    };
    let done = runtime.generate(&[1], &stop, &cancel, |_| {
        Err(std::io::Error::other("stop token must not stream").into())
    })?;
    assert_eq!(done.finish, Finish::Stop);
    assert_eq!(done.generated_tokens, 0);
    assert_eq!(runtime.tokens(), &[1]);
    Ok(())
}

#[test]
fn snapshot_budget_rejects_before_mutating_a_committed_conversation() -> Result<()> {
    let decoder = decoder();
    let steps = decoder.steps.clone();
    let mut runtime = Runtime::new(
        decoder,
        Limits {
            checkpoint_bytes: 1,
            ..LIMITS
        },
    )?;
    assert!(
        runtime
            .generate(&[1], &options(), &Cancellation::default(), |_| Ok(()))
            .is_err()
    );
    assert_eq!(steps.get(), 0);
    assert_eq!(runtime.tokens(), &[0_u32; 0]);
    Ok(())
}

#[test]
fn failed_optional_checkpoint_preserves_a_full_prefix_cache() -> Result<()> {
    let decoder = decoder();
    let checkpoint_fail = decoder.checkpoint_fail.clone();
    let mut runtime = Runtime::new(
        decoder,
        Limits {
            checkpoints: 1,
            ..LIMITS
        },
    )?;
    let cancellation = Cancellation::default();
    runtime.generate(&[1], &options(), &cancellation, |_| Ok(()))?;
    assert_eq!(runtime.checkpoint_count(), 1);

    runtime.generate(&[1, 1, 2], &options(), &cancellation, |_| {
        // The rollback checkpoint already exists. Fail only the optional
        // replacement captured after the request has completed successfully.
        checkpoint_fail.set(true);
        Ok(())
    })?;
    assert_eq!(runtime.tokens(), &[1, 1, 2, 4]);
    assert_eq!(runtime.checkpoint_count(), 1);
    assert!(!runtime.is_faulted());

    let edited = runtime.generate(&[1, 1, 3], &options(), &cancellation, |_| Ok(()))?;
    assert_eq!(edited.reused_tokens, 2);
    assert_eq!(runtime.tokens(), &[1, 1, 3, 5]);
    assert_eq!(runtime.checkpoint_count(), 1);
    Ok(())
}

#[test]
fn oversized_optional_checkpoint_preserves_an_older_prefix() -> Result<()> {
    let decoder = decoder();
    let checkpoint_fail = decoder.checkpoint_fail.clone();
    // Charge the state/token/media_spans Vecs, shared-logits Vec and Arc
    // pointer/counts, plus the snapshot byte count. Every snapshot still charges
    // its full token and logits payload despite sharing. Allow the long rollback
    // and small prefix together, but not the medium rollback and a long
    // replacement.
    let snapshot_metadata = 3 * size_of::<Vec<u32>>()
        + size_of::<Vec<bonsai_prism_runtime::runtime::MediaSpan>>()
        + 4 * size_of::<usize>();
    let mut runtime = Runtime::new(
        decoder,
        Limits {
            checkpoints: 1,
            checkpoint_bytes: 2 * snapshot_metadata + 340,
            ..LIMITS
        },
    )?;
    let cancellation = Cancellation::default();
    runtime.generate(&[1], &options(), &cancellation, |_| Ok(()))?;
    runtime.generate(&[1; 9], &options(), &cancellation, |_| {
        // Keep the small prefix while establishing a larger committed state.
        checkpoint_fail.set(true);
        Ok(())
    })?;
    assert_eq!(runtime.tokens().len(), 10);
    assert_eq!(runtime.checkpoint_count(), 1);

    runtime.generate(&[1; 28], &options(), &cancellation, |_| Ok(()))?;
    assert_eq!(runtime.tokens().len(), 29);
    assert_eq!(runtime.checkpoint_count(), 1);

    let edited = runtime.generate(&[1, 1, 3], &options(), &cancellation, |_| Ok(()))?;
    assert_eq!(edited.reused_tokens, 2);
    assert_eq!(runtime.tokens(), &[1, 1, 3, 5]);
    assert_eq!(runtime.checkpoint_count(), 1);
    Ok(())
}

#[derive(Clone, Copy, Default)]
enum OutputFault {
    #[default]
    None,
    PartialFailure,
    WrongSize,
    Nonfinite,
    Cancel,
}

struct Reusing {
    inner: Recurrent,
    storage: Rc<RefCell<Vec<(usize, usize)>>>,
    owned_calls: Rc<Cell<usize>>,
    fault: Rc<Cell<OutputFault>>,
    cancellation: Cancellation,
}

impl Decoder for Reusing {
    type Checkpoint = Vec<u32>;

    fn identity(&self) -> &str {
        self.inner.identity()
    }
    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }
    fn checkpoint_bytes(&self) -> usize {
        self.inner.checkpoint_bytes()
    }
    fn reset(&mut self) -> Result<()> {
        self.inner.reset()
    }
    fn checkpoint(&self) -> Result<Self::Checkpoint> {
        self.inner.checkpoint()
    }
    fn restore(&mut self, checkpoint: &Self::Checkpoint) -> Result<()> {
        self.inner.restore(checkpoint)
    }
    fn evaluate(&mut self, token: u32) -> Result<Vec<f32>> {
        self.owned_calls.set(self.owned_calls.get() + 1);
        let mut output = Vec::new();
        self.evaluate_into(token, &mut output)?;
        Ok(output)
    }
    fn evaluate_into(&mut self, token: u32, output: &mut Vec<f32>) -> Result<()> {
        self.inner.tokens.push(token);
        output.clear();
        output.try_reserve_exact(8)?;
        output.resize(8, -1.0);
        let next = usize::try_from(self.inner.tokens.iter().sum::<u32>() % 8)?;
        output[next] = 1.0;
        self.storage
            .borrow_mut()
            .push((output.as_ptr().addr(), output.capacity()));
        // Deliberately violate the output contract to verify the runtime's
        // defensive validation and rollback independently of the decoder.
        match self.fault.replace(OutputFault::None) {
            OutputFault::None => {}
            OutputFault::PartialFailure => {
                output.truncate(1);
                return Err(std::io::Error::other("partial output after mutation").into());
            }
            OutputFault::WrongSize => {
                output.pop();
            }
            OutputFault::Nonfinite => {
                output[0] = f32::NAN;
            }
            OutputFault::Cancel => self.cancellation.cancel(),
        }
        Ok(())
    }
}

#[test]
fn finite_decoder_default_checks_custom_outputs_and_clears_failures() -> Result<()> {
    let fault = Rc::new(Cell::new(OutputFault::None));
    let mut decoder = Reusing {
        inner: decoder(),
        storage: Rc::new(RefCell::new(Vec::new())),
        owned_calls: Rc::new(Cell::new(0)),
        fault: fault.clone(),
        cancellation: Cancellation::default(),
    };
    let mut output = Vec::with_capacity(8);
    let allocation = output.as_ptr();
    let checked = decoder.evaluate_finite_into(1, &mut output)?;
    assert_eq!(
        checked.as_slice(),
        &[-1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
    );
    assert_eq!(checked.as_slice().as_ptr(), allocation);
    for failure in [
        OutputFault::PartialFailure,
        OutputFault::WrongSize,
        OutputFault::Nonfinite,
    ] {
        fault.set(failure);
        assert!(decoder.evaluate_finite_into(2, &mut output).is_err());
        assert_eq!(output, Vec::<f32>::new());
        assert_eq!(output.as_ptr(), allocation);
    }
    Ok(())
}

struct FiniteDecoder {
    inner: Recurrent,
    calls: Rc<Cell<usize>>,
    forged: Rc<Cell<usize>>,
}
impl Decoder for FiniteDecoder {
    type Checkpoint = Vec<u32>;

    fn identity(&self) -> &str {
        self.inner.identity()
    }
    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }
    fn checkpoint_bytes(&self) -> usize {
        self.inner.checkpoint_bytes()
    }
    fn reset(&mut self) -> Result<()> {
        self.inner.reset()
    }
    fn checkpoint(&self) -> Result<Self::Checkpoint> {
        self.inner.checkpoint()
    }
    fn restore(&mut self, checkpoint: &Self::Checkpoint) -> Result<()> {
        self.inner.restore(checkpoint)
    }
    fn evaluate(&mut self, _: u32) -> Result<Vec<f32>> {
        Err(std::io::Error::other("runtime must request the finite output").into())
    }
    fn evaluate_finite_into<'a>(
        &mut self,
        token: u32,
        output: &'a mut Vec<f32>,
    ) -> Result<FiniteLogits<'a>> {
        self.calls.set(self.calls.get() + 1);
        *output = self.inner.evaluate(token)?;
        match self.forged.replace(0) {
            1 => {
                output[0] = f32::NAN;
                return FiniteLogits::new(&[0.0; 8]);
            }
            2 => return FiniteLogits::new(&output[..7]),
            3 => {
                output.push(f32::NAN);
                return FiniteLogits::new(&output[..8]);
            }
            _ => {}
        }
        FiniteLogits::new(output)
    }
}
#[test]
fn runtime_consumes_finite_decoder_output_and_rejects_unrelated_proofs() -> Result<()> {
    let calls = Rc::new(Cell::new(0));
    let forged = Rc::new(Cell::new(0));
    let mut runtime = Runtime::new(
        FiniteDecoder {
            inner: decoder(),
            calls: calls.clone(),
            forged: forged.clone(),
        },
        LIMITS,
    )?;
    let cancellation = Cancellation::default();
    runtime.generate(&[1], &options(), &cancellation, |_| Ok(()))?;
    assert_eq!(calls.get(), 2);
    assert_eq!(runtime.tokens(), &[1, 1]);
    let delivered = Cell::new(0);
    for proof in 1..=3 {
        forged.set(proof);
        assert!(
            runtime
                .generate(&[2], &options(), &cancellation, |_| {
                    delivered.set(delivered.get() + 1);
                    Ok(())
                })
                .is_err()
        );
        assert_eq!(calls.get(), 2 + proof);
        assert_eq!(delivered.get(), 0);
        assert_eq!(runtime.tokens(), &[1, 1]);
        assert!(!runtime.is_faulted());
    }
    let completion = runtime.generate(&[1, 1], &options(), &cancellation, |_| Ok(()))?;
    assert_eq!(completion.reused_tokens, 2);
    assert_eq!(calls.get(), 6);
    assert_eq!(runtime.tokens(), &[1, 1, 2]);
    Ok(())
}

#[test]
fn validated_sampler_never_receives_invalid_decoder_outputs_and_rolls_back() -> Result<()> {
    #[derive(Clone)]
    struct ValidatedSampler {
        draws: usize,
        calls: Rc<Cell<usize>>,
        poison: Rc<Cell<bool>>,
        fault: Rc<Cell<OutputFault>>,
    }
    impl Sampler for ValidatedSampler {
        fn sample(&mut self, _: &[f32]) -> Result<u32> {
            Err(std::io::Error::other("runtime must supply its validated view").into())
        }
        fn sample_finite(&mut self, logits: FiniteLogits<'_>) -> Result<u32> {
            assert!(logits.as_slice().iter().all(|value| value.is_finite()));
            self.draws += 1;
            self.calls.set(self.calls.get() + 1);
            if self.poison.replace(false) {
                self.fault.set(OutputFault::Nonfinite);
            }
            bonsai_prism_runtime::runtime::greedy(logits.as_slice())
        }
    }
    let fault = Rc::new(Cell::new(OutputFault::None));
    let poison = Rc::new(Cell::new(false));
    let calls = Rc::new(Cell::new(0));
    let mut sampler = ValidatedSampler {
        draws: 0,
        calls: calls.clone(),
        poison: poison.clone(),
        fault: fault.clone(),
    };
    let cancellation = Cancellation::default();
    let mut runtime = Runtime::new(
        Reusing {
            inner: decoder(),
            storage: Rc::new(RefCell::new(Vec::new())),
            owned_calls: Rc::new(Cell::new(0)),
            fault: fault.clone(),
            cancellation: cancellation.clone(),
        },
        LIMITS,
    )?;
    runtime.generate_sampled(
        &[1],
        &options(),
        Prefill::default(),
        &cancellation,
        &mut sampler,
        |_| Ok(()),
    )?;
    assert_eq!(runtime.tokens(), &[1, 1]);
    assert_eq!(sampler.draws, 1);
    // Invalid prefill output must fail before the typed sampler is invoked.
    fault.set(OutputFault::Nonfinite);
    assert!(
        runtime
            .generate_sampled(
                &[2],
                &options(),
                Prefill::default(),
                &cancellation,
                &mut sampler,
                |_| Ok(())
            )
            .is_err()
    );
    assert_eq!(calls.get(), 1);
    assert_eq!(sampler.draws, 1);
    assert_eq!(runtime.tokens(), &[1, 1]);
    // A valid saved prefix may be sampled, but its following invalid decoder
    // output must roll back both the private logits and caller-owned draws.
    poison.set(true);
    assert!(
        runtime
            .generate_sampled(
                &[1, 1],
                &options(),
                Prefill::default(),
                &cancellation,
                &mut sampler,
                |_| Ok(())
            )
            .is_err()
    );
    assert_eq!(calls.get(), 2);
    assert_eq!(sampler.draws, 1);
    assert_eq!(runtime.tokens(), &[1, 1]);
    assert!(!runtime.is_faulted());
    let completion = runtime.generate_sampled(
        &[1, 1],
        &options(),
        Prefill::default(),
        &cancellation,
        &mut sampler,
        |_| Ok(()),
    )?;
    assert_eq!(completion.reused_tokens, 2);
    assert_eq!(runtime.tokens(), &[1, 1, 2]);
    assert_eq!(sampler.draws, 2);
    assert_eq!(calls.get(), 3);
    Ok(())
}

#[test]
fn reusable_logits_preserve_storage_prefixes_and_transaction_recovery() -> Result<()> {
    let storage = Rc::new(RefCell::new(Vec::new()));
    let owned_calls = Rc::new(Cell::new(0));
    let fault = Rc::new(Cell::new(OutputFault::None));
    let cancellation = Cancellation::default();
    let mut runtime = Runtime::new(
        Reusing {
            inner: decoder(),
            storage: storage.clone(),
            owned_calls: owned_calls.clone(),
            fault: fault.clone(),
            cancellation: cancellation.clone(),
        },
        LIMITS,
    )?;
    let active = Cancellation::default();
    runtime.generate(&[1], &options(), &active, |_| Ok(()))?;
    runtime.generate(&[1, 1, 2], &options(), &active, |_| Ok(()))?;
    let edited = runtime.generate(&[1, 1, 3], &options(), &active, |_| Ok(()))?;
    assert_eq!(edited.reused_tokens, 2);
    assert_eq!(runtime.tokens(), &[1, 1, 3, 5]);
    runtime.reset()?;
    runtime.generate(&[1], &options(), &active, |_| Ok(()))?;
    let captured = storage.borrow();
    // Retained immutable snapshots keep their own distribution. Each request
    // detaches once, then all of its decoder steps reuse the unique allocation.
    assert_eq!(captured.len(), 8);
    let (pairs, remainder) = captured.as_chunks::<2>();
    assert_eq!(remainder.len(), 0);
    for pair in pairs {
        assert_eq!(pair[0], pair[1]);
        assert_eq!(pair[0].1, 8);
    }
    assert_ne!(captured[0].0, captured[2].0);
    assert_ne!(captured[2].0, captured[4].0);
    // Explicit reset discards snapshots and retains the now-unique allocation.
    assert_eq!(captured[4], captured[6]);
    drop(captured);
    assert_eq!(owned_calls.get(), 0);

    for failure in [
        OutputFault::PartialFailure,
        OutputFault::WrongSize,
        OutputFault::Nonfinite,
    ] {
        fault.set(failure);
        assert!(
            runtime
                .generate(&[2], &options(), &active, |_| Ok(()))
                .is_err()
        );
        assert_eq!(runtime.tokens(), &[1, 1]);
        assert!(!runtime.is_faulted());
        // Reusing the exact committed prefix must use its saved logits rather
        // than the partial/invalid buffer written by the failed evaluation.
        let restored = runtime.generate(&[1, 1], &options(), &active, |_| Ok(()))?;
        assert_eq!(restored.reused_tokens, 2);
        assert_eq!(runtime.tokens(), &[1, 1, 2]);
        runtime.reset()?;
        runtime.generate(&[1], &options(), &active, |_| Ok(()))?;
    }
    fault.set(OutputFault::Cancel);
    assert!(
        runtime
            .generate(&[2], &options(), &cancellation, |_| Ok(()))
            .is_err()
    );
    assert_eq!(runtime.tokens(), &[1, 1]);
    assert!(!runtime.is_faulted());
    runtime.generate(&[1, 1], &options(), &active, |_| Ok(()))?;
    assert_eq!(runtime.tokens(), &[1, 1, 2]);
    assert_eq!(owned_calls.get(), 0);
    Ok(())
}

#[test]
fn owned_decoder_fallback_reuses_output_and_clears_failed_evaluations() -> Result<()> {
    let mut decoder = decoder();
    let mut output = Vec::with_capacity(16);
    decoder.evaluate_into(1, &mut output)?;
    let address = output.as_ptr();
    let checkpoint = decoder.checkpoint()?;
    let expected = decoder.evaluate(2)?;
    decoder.restore(&checkpoint)?;
    decoder.evaluate_into(2, &mut output)?;
    assert_eq!(output, expected);
    assert_eq!(output.as_ptr(), address);
    decoder.fail.set(true);
    assert!(decoder.evaluate_into(3, &mut output).is_err());
    assert_eq!(output, [0.0_f32; 0]);
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Operation {
    Ingest(u32),
    Evaluate(u32),
}

#[derive(Clone, Copy, Default)]
enum IngestFault {
    #[default]
    None,
    Failure,
    Cancel,
}

struct StateOnly {
    inner: Reusing,
    operations: Rc<RefCell<Vec<Operation>>>,
    fault: Rc<Cell<IngestFault>>,
    fault_token: Rc<Cell<Option<u32>>>,
}

impl Decoder for StateOnly {
    type Checkpoint = Vec<u32>;

    fn identity(&self) -> &str {
        self.inner.identity()
    }
    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }
    fn checkpoint_bytes(&self) -> usize {
        self.inner.checkpoint_bytes()
    }
    fn reset(&mut self) -> Result<()> {
        self.inner.reset()
    }
    fn checkpoint(&self) -> Result<Self::Checkpoint> {
        self.inner.checkpoint()
    }
    fn restore(&mut self, checkpoint: &Self::Checkpoint) -> Result<()> {
        self.inner.restore(checkpoint)
    }
    fn evaluate(&mut self, token: u32) -> Result<Vec<f32>> {
        self.inner.evaluate(token)
    }
    fn evaluate_into(&mut self, token: u32, output: &mut Vec<f32>) -> Result<()> {
        self.operations
            .borrow_mut()
            .push(Operation::Evaluate(token));
        self.inner.evaluate_into(token, output)
    }
    fn ingest(&mut self, token: u32) -> Result<()> {
        self.operations.borrow_mut().push(Operation::Ingest(token));
        self.inner.inner.tokens.push(token);
        let fault = if self.fault_token.get().is_none_or(|target| target == token) {
            self.fault.replace(IngestFault::None)
        } else {
            IngestFault::None
        };
        match fault {
            IngestFault::None => {}
            IngestFault::Failure => {
                return Err(std::io::Error::other("ingestion failed after state mutation").into());
            }
            IngestFault::Cancel => self.inner.cancellation.cancel(),
        }
        Ok(())
    }
}

fn state_only() -> StateOnly {
    StateOnly {
        inner: Reusing {
            inner: decoder(),
            storage: Rc::new(RefCell::new(Vec::new())),
            owned_calls: Rc::new(Cell::new(0)),
            fault: Rc::new(Cell::new(OutputFault::None)),
            cancellation: Cancellation::default(),
        },
        operations: Rc::new(RefCell::new(Vec::new())),
        fault: Rc::new(Cell::new(IngestFault::None)),
        fault_token: Rc::new(Cell::new(None)),
    }
}

#[test]
fn prompt_ingestion_skips_only_intermediate_uncached_logits_and_preserves_storage() -> Result<()> {
    use Operation::{Evaluate, Ingest};

    let decoder = state_only();
    let operations = decoder.operations.clone();
    let storage = decoder.inner.storage.clone();
    let owned_calls = decoder.inner.owned_calls.clone();
    let mut runtime = Runtime::new(decoder, LIMITS)?;
    let cancellation = Cancellation::default();
    let first = runtime.generate(
        &[1, 2, 3],
        &Generation {
            max_tokens: 2,
            stop_tokens: vec![],
        },
        &cancellation,
        |_| Ok(()),
    )?;
    assert_eq!(first.reused_tokens, 0);
    assert_eq!(runtime.tokens(), &[1, 2, 3, 6, 4]);
    assert_eq!(
        operations.replace(Vec::new()),
        [Ingest(1), Ingest(2), Evaluate(3), Evaluate(6), Evaluate(4)]
    );
    let cached = runtime.generate(&[1, 2, 3, 6, 4], &options(), &cancellation, |_| Ok(()))?;
    assert_eq!(cached.reused_tokens, 5);
    assert_eq!(operations.replace(Vec::new()), [Evaluate(0)]);
    let extended =
        runtime.generate(&[1, 2, 3, 6, 4, 0, 2, 1], &options(), &cancellation, |_| {
            Ok(())
        })?;
    assert_eq!(extended.reused_tokens, 6);
    assert_eq!(
        operations.replace(Vec::new()),
        [Ingest(2), Evaluate(1), Evaluate(3)]
    );
    let edited = runtime.generate(
        &[1, 2, 3, 6, 4, 7, 1],
        &options(),
        &cancellation,
        |_| Ok(()),
    )?;
    assert_eq!(edited.reused_tokens, 5);
    assert_eq!(runtime.tokens(), &[1, 2, 3, 6, 4, 7, 1, 0]);
    assert_eq!(
        operations.replace(Vec::new()),
        [Ingest(7), Evaluate(1), Evaluate(0)]
    );
    let captured = storage.borrow();
    // Each request's immutable rollback/completion distributions stay alive;
    // all evaluated tokens within one request reuse the detached allocation.
    assert_eq!(captured.len(), 8);
    for [start, end] in [[0, 3], [3, 4], [4, 6], [6, 8]] {
        let first = captured[start];
        assert_eq!(first.1, 8);
        assert!(captured[start..end].iter().all(|&entry| entry == first));
    }
    assert_ne!(captured[0].0, captured[3].0);
    assert_ne!(captured[3].0, captured[4].0);
    assert_ne!(captured[4].0, captured[6].0);
    drop(captured);
    assert_eq!(owned_calls.get(), 0);
    Ok(())
}

#[test]
fn ingestion_failure_and_cancellation_restore_committed_state_and_logits() -> Result<()> {
    for failure in [IngestFault::Failure, IngestFault::Cancel] {
        let decoder = state_only();
        let operations = decoder.operations.clone();
        let fault = decoder.fault.clone();
        let cancellation = decoder.inner.cancellation.clone();
        let mut runtime = Runtime::new(decoder, LIMITS)?;
        runtime.generate(&[1], &options(), &cancellation, |_| Ok(()))?;
        operations.borrow_mut().clear();
        fault.set(failure);
        assert!(
            runtime
                .generate(&[2, 3, 4], &options(), &cancellation, |_| Ok(()))
                .is_err()
        );
        assert_eq!(operations.replace(Vec::new()), [Operation::Ingest(2)]);
        assert_eq!(runtime.tokens(), &[1, 1]);
        assert!(!runtime.is_faulted());
        let cancelled = Cancellation::default();
        cancelled.cancel();
        assert!(
            runtime
                .generate(&[2, 3, 4], &options(), &cancelled, |_| Ok(()))
                .is_err()
        );
        assert!(operations.borrow().is_empty());
        let recovered =
            runtime.generate(&[1, 1], &options(), &Cancellation::default(), |_| Ok(()))?;
        assert_eq!(recovered.reused_tokens, 2);
        assert_eq!(runtime.tokens(), &[1, 1, 2]);
        assert_eq!(operations.replace(Vec::new()), [Operation::Evaluate(2)]);
    }
    Ok(())
}

#[test]
fn failed_ingestion_restoration_latches_admission_until_reset() -> Result<()> {
    let decoder = state_only();
    let fault = decoder.fault.clone();
    let restore_fail = decoder.inner.inner.restore_fail.clone();
    let operations = decoder.operations.clone();
    let mut runtime = Runtime::new(decoder, LIMITS)?;
    let cancellation = Cancellation::default();
    runtime.generate(&[1], &options(), &cancellation, |_| Ok(()))?;
    operations.borrow_mut().clear();
    fault.set(IngestFault::Failure);
    restore_fail.set(true);
    assert!(
        runtime
            .generate(&[2, 3], &options(), &cancellation, |_| Ok(()))
            .is_err()
    );
    assert!(runtime.is_faulted());
    assert_eq!(runtime.tokens(), &[1, 1]);
    assert_eq!(operations.replace(Vec::new()), [Operation::Ingest(2)]);
    assert!(
        runtime
            .generate(&[1, 1], &options(), &cancellation, |_| Ok(()))
            .is_err()
    );
    assert!(operations.borrow().is_empty());
    runtime.reset()?;
    restore_fail.set(false);
    runtime.generate(&[3, 4], &options(), &cancellation, |_| Ok(()))?;
    assert_eq!(runtime.tokens(), &[3, 4, 7]);
    Ok(())
}

#[test]
fn default_ingestion_rejects_malformed_discarded_logits_and_rolls_back() -> Result<()> {
    for failure in [
        OutputFault::PartialFailure,
        OutputFault::WrongSize,
        OutputFault::Nonfinite,
    ] {
        let decoder = state_only().inner;
        let fault = decoder.fault.clone();
        let owned_calls = decoder.owned_calls.clone();
        let mut runtime = Runtime::new(decoder, LIMITS)?;
        let cancellation = Cancellation::default();
        runtime.generate(&[1], &options(), &cancellation, |_| Ok(()))?;
        fault.set(failure);
        assert!(
            runtime
                .generate(&[2, 3, 4], &options(), &cancellation, |_| Ok(()))
                .is_err()
        );
        assert_eq!(runtime.tokens(), &[1, 1]);
        assert!(!runtime.is_faulted());
        let recovered = runtime.generate(&[1, 1], &options(), &cancellation, |_| Ok(()))?;
        assert_eq!(recovered.reused_tokens, 2);
        assert_eq!(runtime.tokens(), &[1, 1, 2]);
        // A decoder supplying only reusable output remains backward compatible;
        // the default ingestion path must dispatch through that implementation.
        assert_eq!(owned_calls.get(), 0);
    }
    Ok(())
}

#[test]
fn deadline_expired_admission_preserves_committed_state_and_snapshots() -> Result<()> {
    let decoder = decoder();
    let steps = decoder.steps.clone();
    let checkpoint_fail = decoder.checkpoint_fail.clone();
    let mut runtime = Runtime::new(decoder, LIMITS)?;
    let cancellation = Cancellation::default();
    runtime.generate(&[1], &options(), &cancellation, |_| Ok(()))?;
    let completed_steps = steps.get();
    let saved = runtime.checkpoint_count();
    checkpoint_fail.set(true);
    let result = runtime.generate_until(&[2, 3], &options(), &cancellation, Instant::now(), |_| {
        Err(std::io::Error::other("expired admission must not emit").into())
    });
    assert!(result.is_err());
    if let Err(error) = result {
        assert_eq!(
            error
                .downcast_ref::<std::io::Error>()
                .map(std::io::Error::kind),
            Some(std::io::ErrorKind::TimedOut)
        );
    }
    assert_eq!(steps.get(), completed_steps);
    assert_eq!(runtime.checkpoint_count(), saved);
    assert_eq!(runtime.tokens(), &[1, 1]);
    assert!(checkpoint_fail.get(), "expired admission must not capture");
    assert!(!runtime.is_faulted());
    checkpoint_fail.set(false);
    let completion = runtime.generate_until(
        &[1, 1],
        &options(),
        &cancellation,
        Instant::now() + Duration::from_mins(1),
        |_| Ok(()),
    )?;
    assert_eq!(completion.reused_tokens, 2);
    assert_eq!(runtime.tokens(), &[1, 1, 2]);
    Ok(())
}

#[test]
fn deadline_generation_retains_cancellation_precedence_and_stream_rollback() -> Result<()> {
    let mut runtime = Runtime::new(decoder(), LIMITS)?;
    let active = Cancellation::default();
    runtime.generate(&[1], &options(), &active, |_| Ok(()))?;
    let cancelled = Cancellation::default();
    cancelled.cancel();
    let result = runtime.generate_until(&[2], &options(), &cancelled, Instant::now(), |_| Ok(()));
    assert!(result.is_err());
    if let Err(error) = result {
        assert_eq!(
            error
                .downcast_ref::<std::io::Error>()
                .map(std::io::Error::kind),
            Some(std::io::ErrorKind::Interrupted)
        );
    }
    assert!(
        runtime
            .generate_until(
                &[2],
                &options(),
                &active,
                Instant::now() + Duration::from_mins(1),
                |_| {
                    active.cancel();
                    Ok(())
                },
            )
            .is_err()
    );
    assert_eq!(runtime.tokens(), &[1, 1]);
    assert!(!runtime.is_faulted());
    runtime.generate(&[1, 1], &options(), &Cancellation::default(), |_| Ok(()))?;
    assert_eq!(runtime.tokens(), &[1, 1, 2]);
    Ok(())
}

struct Scheduled {
    inner: StateOnly,
    blocks: Vec<Vec<u32>>,
}

impl Decoder for Scheduled {
    type Checkpoint = Vec<u32>;

    fn identity(&self) -> &str {
        self.inner.identity()
    }
    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }
    fn checkpoint_bytes(&self) -> usize {
        self.inner.checkpoint_bytes()
    }
    fn reset(&mut self) -> Result<()> {
        self.inner.reset()
    }
    fn checkpoint(&self) -> Result<Self::Checkpoint> {
        self.inner.checkpoint()
    }
    fn restore(&mut self, checkpoint: &Self::Checkpoint) -> Result<()> {
        self.inner.restore(checkpoint)
    }
    fn evaluate(&mut self, token: u32) -> Result<Vec<f32>> {
        self.inner.evaluate(token)
    }
    fn evaluate_into(&mut self, token: u32, output: &mut Vec<f32>) -> Result<()> {
        self.inner.evaluate_into(token, output)
    }
    fn ingest(&mut self, token: u32) -> Result<()> {
        self.inner.ingest(token)
    }
    fn ingest_block(&mut self, tokens: &[u32], check: &dyn Fn() -> Result<()>) -> Result<()> {
        self.blocks.push(tokens.to_vec());
        // Exercise the actual trait fallback, including checks within a block.
        self.inner.ingest_block(tokens, check)
    }
}

fn scheduled() -> Scheduled {
    Scheduled {
        inner: state_only(),
        blocks: Vec::new(),
    }
}

#[test]
fn bounded_prefill_schedules_causal_blocks_and_tails_with_exact_output_parity() -> Result<()> {
    let limits = Limits {
        context_tokens: 1024,
        checkpoint_bytes: 65_536,
        ..LIMITS
    };
    let generation = Generation {
        max_tokens: 3,
        stop_tokens: vec![],
    };
    for length in [1, 2, 3, 4, 5, 8, 9, 255, 256, 257, 258, 514] {
        let prompt = (0..length)
            .map(|index| u32::try_from(index % 7 + 1))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let mut reference = Runtime::new(state_only(), limits)?;
        let cancellation = Cancellation::default();
        let mut expected = Vec::new();
        let completion = reference.generate(&prompt, &generation, &cancellation, |token| {
            expected.push(token);
            Ok(())
        })?;
        for block_tokens in [1, 2, 3, 4, MAX_PREFILL_BLOCK_TOKENS] {
            let mut runtime = Runtime::new(scheduled(), limits)?;
            let mut actual = Vec::new();
            assert_eq!(
                runtime.generate_with_prefill(
                    &prompt,
                    &generation,
                    Prefill { block_tokens },
                    &cancellation,
                    |token| {
                        actual.push(token);
                        Ok(())
                    },
                )?,
                completion
            );
            assert_eq!(actual, expected);
            assert_eq!(runtime.tokens(), reference.tokens());
            assert_eq!(
                *runtime.decoder().inner.operations.borrow(),
                *reference.decoder().operations.borrow()
            );
            assert_eq!(
                runtime.decoder().blocks,
                prompt[..prompt.len() - 1]
                    .chunks(block_tokens)
                    .map(<[u32]>::to_vec)
                    .collect::<Vec<_>>()
            );
        }
    }
    Ok(())
}

#[test]
fn bounded_prefill_schedules_only_uncached_tokens_after_exact_prefix_restore() -> Result<()> {
    let mut runtime = Runtime::new(scheduled(), LIMITS)?;
    let cancellation = Cancellation::default();
    let prefill = Prefill { block_tokens: 2 };
    runtime.generate_with_prefill(&[1], &options(), prefill, &cancellation, |_| Ok(()))?;
    let extended = runtime.generate_with_prefill(
        &[1, 1, 2, 3, 4, 5],
        &options(),
        prefill,
        &cancellation,
        |_| Ok(()),
    )?;
    assert_eq!(extended.reused_tokens, 2);
    assert_eq!(runtime.decoder().blocks, [vec![2, 3], vec![4]]);
    let edited = runtime.generate_with_prefill(
        &[1, 1, 6, 7, 2, 3],
        &options(),
        prefill,
        &cancellation,
        |_| Ok(()),
    )?;
    assert_eq!(edited.reused_tokens, 2);
    assert_eq!(runtime.tokens(), &[1, 1, 6, 7, 2, 3, 4]);
    assert_eq!(
        runtime.decoder().blocks,
        [vec![2, 3], vec![4], vec![6, 7], vec![2]]
    );
    let committed = runtime.tokens().to_vec();
    let cached = runtime.generate_with_prefill(
        &committed,
        &options(),
        prefill,
        &cancellation,
        |_| Ok(()),
    )?;
    assert_eq!(cached.reused_tokens, committed.len());
    assert_eq!(runtime.decoder().blocks.len(), 4);
    assert_eq!(runtime.tokens().last(), Some(&0));
    Ok(())
}

#[test]
fn bounded_prefill_partial_first_and_later_blocks_roll_back_and_recover() -> Result<()> {
    for failure in [IngestFault::Failure, IngestFault::Cancel] {
        for failed_token in [3, 7] {
            let decoder = scheduled();
            let fault = decoder.inner.fault.clone();
            let fault_token = decoder.inner.fault_token.clone();
            let cancellation = decoder.inner.inner.cancellation.clone();
            let operations = decoder.inner.operations.clone();
            let mut runtime = Runtime::new(decoder, LIMITS)?;
            let prefill = Prefill { block_tokens: 2 };
            runtime.generate(&[1], &options(), &cancellation, |_| Ok(()))?;
            operations.borrow_mut().clear();
            fault_token.set(Some(failed_token));
            fault.set(failure);
            assert!(
                runtime
                    .generate_with_prefill(
                        &[2, 3, 4, 5, 6, 7, 0],
                        &options(),
                        prefill,
                        &cancellation,
                        |_| Err(std::io::Error::other("partial prefill must not emit").into()),
                    )
                    .is_err()
            );
            assert_eq!(
                *operations.borrow(),
                (2..=failed_token)
                    .map(Operation::Ingest)
                    .collect::<Vec<_>>()
            );
            assert_eq!(runtime.tokens(), &[1, 1]);
            assert_eq!(runtime.decoder().inner.inner.inner.tokens, [1, 1]);
            assert!(!runtime.is_faulted());
            let recovered = runtime.generate_with_prefill(
                &[1, 1],
                &options(),
                prefill,
                &Cancellation::default(),
                |_| Ok(()),
            )?;
            assert_eq!(recovered.reused_tokens, 2);
            assert_eq!(runtime.tokens(), &[1, 1, 2]);
        }
    }
    Ok(())
}

#[test]
fn bounded_prefill_invalid_sizes_reject_before_capture_or_decoder_mutation() -> Result<()> {
    let decoder = scheduled();
    let checkpoint_fail = decoder.inner.inner.inner.checkpoint_fail.clone();
    let operations = decoder.inner.operations.clone();
    let mut runtime = Runtime::new(decoder, LIMITS)?;
    let cancellation = Cancellation::default();
    runtime.generate(&[1], &options(), &cancellation, |_| Ok(()))?;
    operations.borrow_mut().clear();
    checkpoint_fail.set(true);
    let checkpoints = runtime.checkpoint_count();
    for block_tokens in [0, MAX_PREFILL_BLOCK_TOKENS + 1, usize::MAX] {
        for prompt in [&[2, 3][..], &[1, 1][..]] {
            assert!(
                runtime
                    .generate_with_prefill(
                        prompt,
                        &options(),
                        Prefill { block_tokens },
                        &cancellation,
                        |_| Ok(()),
                    )
                    .is_err()
            );
            assert!(
                runtime
                    .generate_until_with_prefill(
                        prompt,
                        &options(),
                        Prefill { block_tokens },
                        &cancellation,
                        Instant::now() + Duration::from_mins(1),
                        |_| Ok(()),
                    )
                    .is_err()
            );
            assert!(checkpoint_fail.get());
            assert_eq!(runtime.tokens(), &[1, 1]);
            assert_eq!(runtime.decoder().inner.inner.inner.tokens, [1, 1]);
            assert!(operations.borrow().is_empty());
            assert_eq!(runtime.decoder().blocks, Vec::<Vec<u32>>::new());
            assert_eq!(runtime.checkpoint_count(), checkpoints);
            assert!(!runtime.is_faulted());
        }
    }
    Ok(())
}

#[test]
fn bounded_prefill_fallback_validates_blocks_before_ingestion_and_checks_each_token() -> Result<()>
{
    let mut decoder = decoder();
    let checks = Cell::new(0);
    let check = || {
        checks.set(checks.get() + 1);
        Ok(())
    };
    for block in [vec![], vec![1; MAX_PREFILL_BLOCK_TOKENS + 1], vec![1, 8]] {
        assert!(decoder.ingest_block(&block, &check).is_err());
        assert_eq!(decoder.tokens, [0_u32; 0]);
        assert_eq!(decoder.steps.get(), 0);
        assert_eq!(checks.get(), 0);
    }
    let block = vec![1; MAX_PREFILL_BLOCK_TOKENS];
    decoder.ingest_block(&block, &check)?;
    assert_eq!(decoder.tokens, block);
    assert_eq!(decoder.steps.get(), MAX_PREFILL_BLOCK_TOKENS);
    assert_eq!(checks.get(), 2 * MAX_PREFILL_BLOCK_TOKENS);
    Ok(())
}

#[test]
fn bounded_prefill_final_logits_and_stream_failure_restore_every_completed_block() -> Result<()> {
    for callback_failure in [false, true] {
        let decoder = scheduled();
        let output_fault = decoder.inner.inner.fault.clone();
        let mut runtime = Runtime::new(decoder, LIMITS)?;
        let cancellation = Cancellation::default();
        let prefill = Prefill { block_tokens: 2 };
        runtime.generate(&[1], &options(), &cancellation, |_| Ok(()))?;
        if !callback_failure {
            output_fault.set(OutputFault::WrongSize);
        }
        let mut emitted = Vec::new();
        assert!(
            runtime
                .generate_with_prefill(
                    &[2, 3, 4, 5, 6, 7],
                    &options(),
                    prefill,
                    &cancellation,
                    |token| {
                        emitted.push(token);
                        Err(std::io::Error::other("disconnected block-prefill consumer").into())
                    },
                )
                .is_err()
        );
        assert_eq!(emitted.len(), usize::from(callback_failure));
        assert_eq!(runtime.decoder().blocks, [vec![2, 3], vec![4, 5], vec![6]]);
        assert_eq!(runtime.tokens(), &[1, 1]);
        assert_eq!(runtime.decoder().inner.inner.inner.tokens, [1, 1]);
        assert!(!runtime.is_faulted());
        let recovered =
            runtime
                .generate_with_prefill(&[1, 1], &options(), prefill, &cancellation, |_| Ok(()))?;
        assert_eq!(recovered.reused_tokens, 2);
        assert_eq!(runtime.tokens(), &[1, 1, 2]);
    }
    Ok(())
}

#[test]
fn bounded_prefill_failed_restore_latches_admission_until_reset() -> Result<()> {
    let decoder = scheduled();
    let fault = decoder.inner.fault.clone();
    let fault_token = decoder.inner.fault_token.clone();
    let restore_fail = decoder.inner.inner.inner.restore_fail.clone();
    let mut runtime = Runtime::new(decoder, LIMITS)?;
    let cancellation = Cancellation::default();
    let prefill = Prefill { block_tokens: 2 };
    runtime.generate(&[1], &options(), &cancellation, |_| Ok(()))?;
    fault.set(IngestFault::Failure);
    fault_token.set(Some(3));
    restore_fail.set(true);
    assert!(
        runtime
            .generate_with_prefill(
                &[2, 3, 4, 5],
                &options(),
                prefill,
                &cancellation,
                |_| Ok(()),
            )
            .is_err()
    );
    assert!(runtime.is_faulted());
    assert_eq!(runtime.tokens(), &[1, 1]);
    assert_eq!(runtime.decoder().blocks, [vec![2, 3]]);
    assert!(
        runtime
            .generate_with_prefill(&[1, 1], &options(), prefill, &cancellation, |_| Ok(()),)
            .is_err()
    );
    assert_eq!(runtime.decoder().blocks, [vec![2, 3]]);
    restore_fail.set(false);
    runtime.reset()?;
    assert!(!runtime.is_faulted());
    runtime.generate_with_prefill(&[4, 5, 6], &options(), prefill, &cancellation, |_| Ok(()))?;
    assert_eq!(runtime.tokens(), &[4, 5, 6, 7]);
    Ok(())
}

#[test]
fn bounded_prefill_final_prompt_logits_can_stop_without_evaluating_a_generated_token() -> Result<()>
{
    let mut runtime = Runtime::new(scheduled(), LIMITS)?;
    let stop = Generation {
        max_tokens: 3,
        stop_tokens: vec![1],
    };
    let completion = runtime.generate_until_with_prefill(
        &[2, 3, 4],
        &stop,
        Prefill { block_tokens: 2 },
        &Cancellation::default(),
        Instant::now() + Duration::from_mins(1),
        |_| Err(std::io::Error::other("stop token must not emit").into()),
    )?;
    assert_eq!(completion.finish, Finish::Stop);
    assert_eq!(completion.generated_tokens, 0);
    assert_eq!(runtime.tokens(), &[2, 3, 4]);
    assert_eq!(runtime.decoder().blocks, [vec![2, 3]]);
    assert_eq!(
        *runtime.decoder().inner.operations.borrow(),
        [
            Operation::Ingest(2),
            Operation::Ingest(3),
            Operation::Evaluate(4)
        ]
    );
    Ok(())
}

#[test]
fn marconi_utility_and_lru_eviction_retains_reused_long_prefix_over_short_throwaway_turns()
-> Result<()> {
    let mut runtime = Runtime::new(
        decoder(),
        Limits {
            context_tokens: 64,
            checkpoints: 2,
            checkpoint_bytes: 32_768,
        },
    )?;
    let cancel = Cancellation::default();
    let base_prompt: Vec<u32> = (0_u32..15).map(|idx| idx % 7).collect();
    runtime.generate(&base_prompt, &options(), &cancel, |_| Ok(()))?;
    let c0_tokens = runtime.tokens().to_vec();
    assert_eq!(c0_tokens.len(), 16);

    // Branch A extends C0 and promotes C0 via an exact prefix-cache hit before committing C1.
    let mut branch_a = c0_tokens.clone();
    branch_a.push(5);
    let hit_a = runtime.generate(&branch_a, &options(), &cancel, |_| Ok(()))?;
    assert_eq!(hit_a.reused_tokens, c0_tokens.len());

    // Two unrelated 1-token throwaway turns fill/rotate the 2-slot cache. Old FIFO eviction
    // would discard C0 and leave only [throwaway_1, throwaway_2], yielding 0 reused tokens
    // on a second fork from C0. Marconi FLOP/byte utility + LRU keeps C0 alive.
    runtime.generate(&[6], &options(), &cancel, |_| Ok(()))?;
    runtime.generate(&[5], &options(), &cancel, |_| Ok(()))?;

    let stats_before_rebranch = runtime.prefix_cache_stats();
    assert_eq!(stats_before_rebranch.evictions, 2);
    assert_eq!(stats_before_rebranch.hits, 1);
    assert_eq!(stats_before_rebranch.misses, 3);

    let mut branch_b = c0_tokens.clone();
    branch_b.push(4);
    let hit_b = runtime.generate(&branch_b, &options(), &cancel, |_| Ok(()))?;
    assert_eq!(hit_b.reused_tokens, c0_tokens.len());
    assert_eq!(runtime.prefix_cache_stats().hits, 2);
    assert_eq!(
        runtime.prefix_cache_stats().reused_tokens,
        u64::try_from(c0_tokens.len() * 2)?
    );
    Ok(())
}

mod grammar_speculation {
    use bonsai_prism_runtime::{
        Decoder, Result,
        grammar::{
            Grammar,
            runtime::{Candidate, GrammarConstrainedSampler, GrammarState},
        },
        runtime::{Cancellation, Finish, Generation, Limits, Prefill, Runtime},
        sampling::{ReferenceSampler, Sampler, Sampling},
        speculative::SuffixDraftConfig,
    };
    use std::{
        cell::Cell,
        rc::Rc,
        sync::Arc,
        time::{Duration, Instant},
    };

    #[derive(Clone, Debug, Default, Eq, PartialEq)]
    struct HybridState {
        tokens: Vec<u32>,
        recurrent: u64,
        conv: [u32; 3],
    }

    #[derive(Clone, Copy, Debug)]
    enum LogitMode {
        FollowCycle,
        FavorIllegalAtStart { illegal: u32, fallback: u32 },
        StopAtStep { step: usize, stop: u32, normal: u32 },
    }

    struct GrammarBatchDecoder {
        state: HybridState,
        capacity: usize,
        batches: usize,
        restores: usize,
        mode: LogitMode,
        cancel_after_evals: Rc<Cell<Option<usize>>>,
        cancellation: Cancellation,
        evals: usize,
    }

    impl GrammarBatchDecoder {
        fn new(capacity: usize, mode: LogitMode, cancellation: Cancellation) -> Self {
            Self {
                state: HybridState::default(),
                capacity,
                batches: 0,
                restores: 0,
                mode,
                cancel_after_evals: Rc::new(Cell::new(None)),
                cancellation,
                evals: 0,
            }
        }

        fn logits_for_state(&self) -> Vec<f32> {
            let mut logits = vec![-6.0_f32; 8];
            match self.mode {
                LogitMode::FollowCycle => {
                    // Tokens after prompt [0, 2, 3, 2, 1, 0]: repeat [2, 3, 2, 1], then EOG 7.
                    let idx = self.state.tokens.len();
                    let preferred = match idx {
                        6 | 8 => 2,
                        7 => 3,
                        9 => 1,
                        _ => 7,
                    };
                    logits[usize::try_from(preferred).unwrap_or(0)] = 6.0;
                    logits[2] = logits[2].max(-1.0);
                    logits[3] = logits[3].max(-1.2);
                }
                LogitMode::FavorIllegalAtStart { illegal, fallback } => {
                    logits[7] = 25.0;
                    logits[usize::try_from(illegal).unwrap_or(0)] = 20.0;
                    logits[usize::try_from(fallback).unwrap_or(0)] = 5.0;
                    logits[1] = -0.5;
                }
                LogitMode::StopAtStep { step, stop, normal } => {
                    let chosen = if self.state.tokens.len() >= step {
                        stop
                    } else {
                        normal
                    };
                    logits[usize::try_from(chosen).unwrap_or(0)] = 8.0;
                    logits[usize::try_from(normal).unwrap_or(0)] =
                        logits[usize::try_from(normal).unwrap_or(0)].max(1.0);
                }
            }
            logits
        }
    }

    impl Decoder for GrammarBatchDecoder {
        type Checkpoint = HybridState;

        fn identity(&self) -> &'static str {
            "grammar-speculative-hybrid/v1"
        }

        fn vocab_size(&self) -> usize {
            8
        }

        fn checkpoint_bytes(&self) -> usize {
            256
        }

        fn reset(&mut self) -> Result<()> {
            self.state = HybridState::default();
            self.batches = 0;
            self.restores = 0;
            self.evals = 0;
            Ok(())
        }

        fn evaluate(&mut self, token: u32) -> Result<Vec<f32>> {
            self.evals += 1;
            if let Some(limit) = self.cancel_after_evals.get()
                && self.evals >= limit
            {
                self.cancellation.cancel();
            }
            self.state.tokens.push(token);
            self.state.recurrent = self
                .state
                .recurrent
                .wrapping_mul(31)
                .wrapping_add(u64::from(token) + 3);
            self.state.conv.rotate_left(1);
            self.state.conv[2] = token;
            Ok(self.logits_for_state())
        }

        fn batched_evaluation_capacity(&self) -> usize {
            self.capacity
        }

        fn evaluate_batch_into(
            &mut self,
            tokens: &[u32],
            rows: &mut Vec<Vec<f32>>,
            check: &dyn Fn() -> Result<()>,
        ) -> Result<()> {
            self.batches += 1;
            rows.clear();
            for &token in tokens {
                check()?;
                rows.push(self.evaluate(token)?);
                check()?;
            }
            Ok(())
        }

        fn checkpoint(&self) -> Result<Self::Checkpoint> {
            Ok(self.state.clone())
        }

        fn restore(&mut self, checkpoint: &Self::Checkpoint) -> Result<()> {
            self.restores += 1;
            self.state = checkpoint.clone();
            Ok(())
        }
    }

    fn vocab_candidates() -> Result<Arc<[Candidate]>> {
        let pieces: [&[u8]; 8] = [b"{", b"}", b"a", b"b", b":", b"\"", b"1", b""];
        let vec = pieces
            .iter()
            .enumerate()
            .map(|(id, piece)| Candidate::from_piece(u32::try_from(id)?, piece))
            .collect::<Result<Vec<_>>>()?;
        Ok(Arc::from(vec))
    }

    fn runtime(
        capacity: usize,
        mode: LogitMode,
        cancellation: Cancellation,
    ) -> Result<Runtime<GrammarBatchDecoder>> {
        Runtime::new(
            GrammarBatchDecoder::new(capacity, mode, cancellation),
            Limits {
                context_tokens: 64,
                checkpoints: 2,
                checkpoint_bytes: 32_768,
            },
        )
    }

    fn constrained_ref(
        grammar_src: &str,
        seed: u64,
        temperature: f32,
        eog: &[u32],
    ) -> Result<GrammarConstrainedSampler<ReferenceSampler>> {
        let grammar = GrammarState::from_grammar(Grammar::parse(grammar_src)?, "root")?;
        let inner = ReferenceSampler::new(
            Sampling {
                temperature,
                top_k: 4,
                top_p: 0.95,
            },
            seed,
        )?;
        GrammarConstrainedSampler::new(inner, grammar, vocab_candidates()?, Arc::from(eog))
    }

    fn assert_rng_continuation_matches(
        left: &mut ReferenceSampler,
        right: &mut ReferenceSampler,
    ) -> Result<()> {
        let probe = [0.2_f32, -0.4, 1.1, 0.0, -1.5, 0.7, 0.3, -0.9];
        for _ in 0..8 {
            assert_eq!(left.sample(&probe)?, right.sample(&probe)?);
        }
        Ok(())
    }

    struct ParityCase<'a> {
        name: &'a str,
        grammar: &'a str,
        mode: LogitMode,
        max_tokens: usize,
        stop_tokens: &'a [u32],
        eog_tokens: &'a [u32],
        expect_finish: Finish,
        expect_accepted: usize,
        expect_full_accept: bool,
        expect_rollback: bool,
    }

    #[derive(Clone, Copy)]
    enum FailureKind {
        CallbackDisconnect,
        CancelInBatch,
        DeadEndGrammar,
    }

    fn verify_parity_case(
        case: &ParityCase<'_>,
        seed: u64,
        prompt: &[u32],
        suffix: SuffixDraftConfig,
        deadline: Instant,
    ) -> Result<()> {
        let mut ordinary = runtime(8, case.mode, Cancellation::default())?;
        let mut speculative = runtime(8, case.mode, Cancellation::default())?;
        let mut seq_sampler = constrained_ref(case.grammar, seed, 0.25, case.eog_tokens)?;
        let mut spec_sampler = seq_sampler.clone();
        let options = Generation {
            max_tokens: case.max_tokens,
            stop_tokens: case.stop_tokens.to_vec(),
        };
        let mut seq_out = Vec::new();
        let mut spec_out = Vec::new();
        let seq_completion = ordinary.generate_sampled_until(
            prompt,
            &options,
            Prefill::default(),
            &Cancellation::default(),
            deadline,
            &mut seq_sampler,
            |token| {
                seq_out.push(token);
                Ok(())
            },
        )?;
        let (spec_completion, stats) = speculative.generate_sampled_speculative_until(
            prompt,
            &options,
            Prefill::default(),
            suffix,
            &Cancellation::default(),
            deadline,
            &mut spec_sampler,
            |token| {
                spec_out.push(token);
                Ok(())
            },
        )?;

        assert_eq!(spec_completion, seq_completion, "case {}", case.name);
        assert_eq!(
            spec_completion.finish, case.expect_finish,
            "case {}",
            case.name
        );
        assert_eq!(spec_out, seq_out, "case {}", case.name);
        assert_eq!(
            speculative.tokens(),
            ordinary.tokens(),
            "case {}",
            case.name
        );
        assert_eq!(
            speculative.decoder().state,
            ordinary.decoder().state,
            "case {}",
            case.name
        );
        assert_eq!(
            spec_sampler.grammar().stacks(),
            seq_sampler.grammar().stacks(),
            "case {}",
            case.name
        );
        assert_eq!(
            spec_sampler.grammar().partial_utf8(),
            seq_sampler.grammar().partial_utf8(),
            "case {}",
            case.name
        );
        assert_eq!(
            spec_sampler.grammar().is_accepting(),
            seq_sampler.grammar().is_accepting(),
            "case {}",
            case.name
        );
        assert_rng_continuation_matches(spec_sampler.inner_mut(), seq_sampler.inner_mut())?;
        assert!(stats.draft_rounds > 0, "case {}", case.name);
        assert!(speculative.decoder().batches > 0, "case {}", case.name);
        assert_eq!(
            stats.draft_tokens_accepted, case.expect_accepted,
            "case {}",
            case.name
        );
        if case.expect_full_accept {
            assert_eq!(
                stats.draft_tokens_accepted, stats.draft_tokens_proposed,
                "case {}",
                case.name
            );
        } else {
            assert!(
                stats.draft_tokens_proposed > stats.draft_tokens_accepted,
                "case {}",
                case.name
            );
        }
        if case.expect_rollback {
            assert!(stats.rollback_restores > 0, "case {}", case.name);
        } else {
            assert_eq!(stats.rollback_restores, 0, "case {}", case.name);
        }
        Ok(())
    }

    fn assert_retry_matches_sequential(
        runtime: &mut Runtime<GrammarBatchDecoder>,
        retry_prompt: &[u32],
        suffix: SuffixDraftConfig,
        deadline: Instant,
        mut retry_sampler: GrammarConstrainedSampler<ReferenceSampler>,
    ) -> Result<()> {
        let mut reference_runtime =
            self::runtime(8, LogitMode::FollowCycle, Cancellation::default())?;
        let mut reference_sampler =
            constrained_ref("root ::= (\"a\" | \"b\")+ \"}\"", 123, 0.6, &[7])?;
        let retry_options = Generation {
            max_tokens: 5,
            stop_tokens: vec![7],
        };
        let mut actual_out = Vec::new();
        let mut expected_out = Vec::new();
        let (actual_completion, _) = runtime.generate_sampled_speculative_until(
            retry_prompt,
            &retry_options,
            Prefill::default(),
            suffix,
            &Cancellation::default(),
            deadline,
            &mut retry_sampler,
            |token| {
                actual_out.push(token);
                Ok(())
            },
        )?;
        let expected_completion = reference_runtime.generate_sampled_until(
            retry_prompt,
            &retry_options,
            Prefill::default(),
            &Cancellation::default(),
            deadline,
            &mut reference_sampler,
            |token| {
                expected_out.push(token);
                Ok(())
            },
        )?;
        assert_eq!(actual_completion, expected_completion);
        assert_eq!(actual_out, expected_out);
        assert_eq!(runtime.tokens(), reference_runtime.tokens());
        assert_eq!(runtime.decoder().state, reference_runtime.decoder().state);
        assert_eq!(
            retry_sampler.grammar().stacks(),
            reference_sampler.grammar().stacks()
        );
        assert_rng_continuation_matches(retry_sampler.inner_mut(), reference_sampler.inner_mut())
    }

    fn verify_rollback_case(
        kind: FailureKind,
        suffix: SuffixDraftConfig,
        deadline: Instant,
    ) -> Result<()> {
        let initial_prompt = [0_u32, 2, 1];
        let retry_prompt = [0_u32, 2, 3, 2, 1, 0];
        let cancellation = Cancellation::default();
        let mut runtime = runtime(8, LogitMode::FollowCycle, cancellation.clone())?;
        let mut base_sampler = constrained_ref("root ::= \"a\" \"}\"", 99, 0.5, &[7])?;
        runtime.generate_sampled_until(
            &[0],
            &Generation {
                max_tokens: 2,
                stop_tokens: vec![7],
            },
            Prefill::default(),
            &Cancellation::default(),
            deadline,
            &mut base_sampler,
            |_| Ok(()),
        )?;
        assert_eq!(runtime.tokens(), &initial_prompt);

        let committed_tokens = runtime.tokens().to_vec();
        let committed_state = runtime.decoder().state.clone();
        let cancel_after = runtime.decoder().cancel_after_evals.clone();
        if matches!(kind, FailureKind::CancelInBatch) {
            // `prepare` resets the decoder (`evals = 0`), then `retry_prompt`
            // evaluates 6 tokens (`evals = 1..=6`), and the 4-token batch
            // evaluates tokens 7..=10, cancelling inside the batch.
            cancel_after.set(Some(8));
        }

        let failing_grammar = match kind {
            FailureKind::DeadEndGrammar => "root ::= \"a\" \"x\"",
            FailureKind::CallbackDisconnect | FailureKind::CancelInBatch => {
                "root ::= (\"a\" | \"b\")+ \"}\""
            }
        };
        let mut sampler = constrained_ref(failing_grammar, 123, 0.6, &[7])?;
        let sampler_snapshot = sampler.clone();

        let mut delivered = 0_usize;
        let err = runtime.generate_sampled_speculative_until(
            &retry_prompt,
            &Generation {
                max_tokens: 4,
                stop_tokens: vec![7],
            },
            Prefill::default(),
            suffix,
            &cancellation,
            deadline,
            &mut sampler,
            |_| {
                delivered += 1;
                if matches!(kind, FailureKind::CallbackDisconnect) && delivered == 2 {
                    return Err(std::io::Error::other("injected disconnect").into());
                }
                Ok(())
            },
        );
        assert!(err.is_err());
        assert!(runtime.decoder().batches > 0);
        if matches!(kind, FailureKind::DeadEndGrammar) {
            assert_eq!(delivered, 1);
        }
        assert!(!runtime.is_faulted());
        assert_eq!(runtime.tokens(), committed_tokens.as_slice());
        assert_eq!(runtime.decoder().state, committed_state);
        assert_eq!(
            sampler.grammar().stacks(),
            sampler_snapshot.grammar().stacks()
        );
        assert_eq!(
            sampler.grammar().partial_utf8(),
            sampler_snapshot.grammar().partial_utf8()
        );
        let mut rng_probe_actual = sampler.inner().clone();
        let mut rng_probe_expected = sampler_snapshot.inner().clone();
        assert_rng_continuation_matches(&mut rng_probe_actual, &mut rng_probe_expected)?;

        cancel_after.set(None);
        let retry_sampler = if matches!(kind, FailureKind::DeadEndGrammar) {
            constrained_ref("root ::= (\"a\" | \"b\")+ \"}\"", 123, 0.6, &[7])?
        } else {
            sampler
        };
        assert_retry_matches_sequential(
            &mut runtime,
            &retry_prompt,
            suffix,
            deadline,
            retry_sampler,
        )
    }

    #[test]
    fn grammar_speculation_matches_sequential_across_accept_partial_full_reject_and_stop()
    -> Result<()> {
        let prompt = [0_u32, 2, 3, 2, 1, 0];
        let suffix = SuffixDraftConfig::new(1, 16, 4)?;
        let deadline = Instant::now() + Duration::from_mins(1);
        let cases = [
            ParityCase {
                name: "full-accept round followed by EOG stop on next round",
                grammar: "root ::= (\"a\" | \"b\")+ \"}\"",
                mode: LogitMode::FollowCycle,
                max_tokens: 8,
                stop_tokens: &[7],
                eog_tokens: &[7],
                expect_finish: Finish::Stop,
                expect_accepted: 4,
                expect_full_accept: false,
                expect_rollback: true,
            },
            ParityCase {
                name: "full-accept exact max_tokens boundary",
                grammar: "root ::= (\"a\" | \"b\")+ \"}\"",
                mode: LogitMode::FollowCycle,
                max_tokens: 4,
                stop_tokens: &[7],
                eog_tokens: &[7],
                expect_finish: Finish::Length,
                expect_accepted: 4,
                expect_full_accept: true,
                expect_rollback: false,
            },
            ParityCase {
                name: "partial-reject enforced by grammar mid-batch",
                grammar: "root ::= \"a\" \"b\" \"b\" \"}\"",
                mode: LogitMode::FollowCycle,
                max_tokens: 5,
                stop_tokens: &[7],
                eog_tokens: &[7],
                expect_finish: Finish::Stop,
                expect_accepted: 2,
                expect_full_accept: false,
                expect_rollback: true,
            },
            ParityCase {
                name: "full-reject with illegal draft and illegal EOG masked out",
                grammar: "root ::= \"b\"+ \"}\"",
                mode: LogitMode::FavorIllegalAtStart {
                    illegal: 2,
                    fallback: 3,
                },
                max_tokens: 2,
                stop_tokens: &[7],
                eog_tokens: &[7],
                expect_finish: Finish::Length,
                expect_accepted: 0,
                expect_full_accept: false,
                expect_rollback: true,
            },
            ParityCase {
                name: "EOG stop sampled inside speculative batch",
                grammar: "root ::= \"a\"+",
                mode: LogitMode::StopAtStep {
                    step: 7,
                    stop: 7,
                    normal: 2,
                },
                max_tokens: 6,
                stop_tokens: &[7],
                eog_tokens: &[7],
                expect_finish: Finish::Stop,
                expect_accepted: 1,
                expect_full_accept: false,
                expect_rollback: true,
            },
        ];

        for case in &cases {
            for seed in [7_u64, 4242] {
                verify_parity_case(case, seed, &prompt, suffix, deadline)?;
            }
        }
        Ok(())
    }

    #[test]
    fn grammar_speculation_rolls_back_state_grammar_and_rng_on_failures_and_retries_identically()
    -> Result<()> {
        let suffix = SuffixDraftConfig::new(1, 16, 4)?;
        let deadline = Instant::now() + Duration::from_mins(1);
        for kind in [
            FailureKind::CallbackDisconnect,
            FailureKind::CancelInBatch,
            FailureKind::DeadEndGrammar,
        ] {
            verify_rollback_case(kind, suffix, deadline)?;
        }
        Ok(())
    }
}
