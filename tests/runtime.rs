#![forbid(unsafe_code)]
//! Recurrent-state ownership regressions independent of a model artifact.

use bonsai_prism_runtime::{
    Decoder, MAX_PREFILL_BLOCK_TOKENS, Result,
    runtime::{Cancellation, Finish, Generation, Limits, Prefill, Runtime},
    sampling::Sampler,
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
        Instant::now() + Duration::from_secs(60),
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
    // Each snapshot owns three Vec headers and a byte count, plus two copies
    // of its token IDs and eight logits. Allow the long rollback and the small
    // saved prefix together, but not the medium rollback and a long replacement.
    let snapshot_metadata = 3 * size_of::<Vec<u32>>() + size_of::<usize>();
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
    let first_storage = storage.borrow()[0];
    assert!(storage.borrow().iter().all(|&entry| entry == first_storage));
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
    let first_storage = storage.borrow()[0];
    assert!(storage.borrow().iter().all(|&entry| entry == first_storage));
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
        Instant::now() + Duration::from_secs(60),
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
                Instant::now() + Duration::from_secs(60),
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
                        Instant::now() + Duration::from_secs(60),
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
        Instant::now() + Duration::from_secs(60),
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
