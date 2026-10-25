//! Transactional token generation, bounded snapshots and exact edited-prefix reuse.

use crate::{
    Decoder, MAX_PREFILL_BLOCK_TOKENS, Result, invalid,
    sampling::{Greedy, Sampler},
};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::time::Instant;

/// Cancellation shared with the request owner; checked between decoder steps.
#[derive(Clone, Default)]
pub struct Cancellation(Arc<AtomicBool>);

impl Cancellation {
    /// Request cancellation; an executing scalar operation completes first.
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Release);
    }

    /// Whether this request has been cancelled.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }

    fn check(&self) -> Result<()> {
        if self.is_cancelled() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Interrupted,
                "native generation cancelled",
            )
            .into());
        }
        Ok(())
    }
}

/// Hard bounds for one token conversation and its independently owned snapshots.
#[derive(Clone, Copy, Debug)]
pub struct Limits {
    /// Maximum prompt plus generated sequence length; no eviction or shifting.
    pub context_tokens: usize,
    /// Number of completed prefixes retained, from zero through eight.
    pub checkpoints: usize,
    /// Total snapshot storage, including the request's rollback snapshot.
    pub checkpoint_bytes: usize,
}

/// Generation bounds and exact stop tokens, independent of sampling policy.
#[derive(Clone, Debug)]
pub struct Generation {
    /// Maximum sampled tokens, including a sampled stop token.
    pub max_tokens: usize,
    /// Exact token IDs that finish generation without entering conversation state.
    pub stop_tokens: Vec<u32>,
}

/// Bounded scheduling for intermediate, uncached prompt tokens.
///
/// Blocks execute causally through [`Decoder::ingest_block`]; the reference
/// fallback retains exact sequential arithmetic. The final prompt token always
/// computes logits separately. This does not select optimized graph kernels or
/// change decode scheduling. Existing generation methods use one-token blocks.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Prefill {
    /// Maximum state-only tokens per block, from one through
    /// [`MAX_PREFILL_BLOCK_TOKENS`]. Invalid sizes reject the request before any
    /// decoder operation or checkpoint eviction.
    pub block_tokens: usize,
}

impl Default for Prefill {
    fn default() -> Self {
        Self { block_tokens: 1 }
    }
}

/// Why a successfully completed request stopped.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Finish {
    /// The sampled next token was in the supplied stop set.
    Stop,
    /// The requested maximum number of sampled tokens was reached.
    Length,
}

/// Accounting for a successful request; no text or private prompt is retained here.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Completion {
    /// Exact original prompt length.
    pub prompt_tokens: usize,
    /// Tokens restored or reused without evaluation.
    pub reused_tokens: usize,
    /// Newly emitted tokens; excludes the stop token.
    pub generated_tokens: usize,
    /// Stop-token or length completion.
    pub finish: Finish,
}

struct Snapshot<S> {
    state: S,
    tokens: Vec<u32>,
    logits: Vec<f32>,
    bytes: usize,
}

/// Single-owner native conversation with rollback on cancellation or failure.
///
/// Edited prompts reuse only an exact saved token prefix. Recurrent state is
/// never truncated as though it were a KV-only cache. A failed stream callback
/// rolls back the entire request, including prefill; already emitted output is
/// provisional until this call succeeds. Failed restoration latches the runtime
/// closed until [`Self::reset`] succeeds. No task or background work is spawned.
pub struct Runtime<D: Decoder> {
    decoder: D,
    limits: Limits,
    tokens: Vec<u32>,
    // Empty means invalid; a valid decoder vocabulary is always nonempty.
    logits: Vec<f32>,
    checkpoints: Vec<Snapshot<D::Checkpoint>>,
    faulted: bool,
}

impl<D: Decoder> Runtime<D> {
    /// Establish an empty conversation and validate all fixed ownership bounds.
    ///
    /// # Errors
    /// Rejects invalid limits, impossible vocabulary sizes, allocation failures,
    /// or a decoder that cannot reset to an empty sequence.
    pub fn new(mut decoder: D, limits: Limits) -> Result<Self> {
        if limits.context_tokens == 0
            || limits.context_tokens > 262_144
            || limits.checkpoints > 8
            || limits.checkpoint_bytes == 0
            || decoder.vocab_size() == 0
            || decoder.vocab_size() > 1_048_576
            || decoder.identity().is_empty()
        {
            return Err(invalid(
                "invalid native conversation limits or decoder identity",
            ));
        }
        decoder.reset()?;
        let mut tokens = Vec::new();
        tokens.try_reserve_exact(limits.context_tokens)?;
        let mut checkpoints = Vec::new();
        checkpoints.try_reserve_exact(limits.checkpoints)?;
        Ok(Self {
            decoder,
            limits,
            tokens,
            logits: Vec::new(),
            checkpoints,
            faulted: false,
        })
    }

    /// Committed prompt plus generated tokens; excludes sampled stop tokens.
    #[must_use]
    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    /// Inspect the decoder's identity and diagnostics without changing ownership.
    ///
    /// Diagnostics describe the backend's latest operation, which may have been
    /// provisional or failed. Consult [`Self::tokens`] for committed history and
    /// [`Self::is_faulted`] before interpreting the backend as reusable state.
    #[must_use]
    pub const fn decoder(&self) -> &D {
        &self.decoder
    }

    /// Whether failed backend restoration or reset requires a successful reset.
    #[must_use]
    pub const fn is_faulted(&self) -> bool {
        self.faulted
    }

    /// Retained complete-prefix snapshot count.
    #[must_use]
    pub const fn checkpoint_count(&self) -> usize {
        self.checkpoints.len()
    }

    /// Drop saved prefixes and clear all model state, recovering a failure latch.
    ///
    /// # Errors
    /// A reset failure leaves admission closed and returns the decoder error.
    /// Committed history and saved prefixes are cleared only after success;
    /// backend state remains unavailable until a subsequent reset succeeds.
    pub fn reset(&mut self) -> Result<()> {
        self.faulted = true;
        self.decoder.reset()?;
        self.checkpoints.clear();
        self.tokens.clear();
        self.logits.clear();
        self.faulted = false;
        Ok(())
    }

    /// Evaluate a nonempty prompt and stream greedy token IDs synchronously.
    ///
    /// The callback supplies backpressure and may return an error to cancel a
    /// disconnected consumer. Context and token validation precede execution.
    /// Cancellation is checked before/after every decoder step and callback.
    ///
    /// # Errors
    /// Rejects invalid tokens/options, resource exhaustion, cancellation,
    /// nonfinite/wrong-size logits, callback errors and decoder failures. State
    /// returns to the previous committed prefix unless restoration itself fails.
    pub fn generate(
        &mut self,
        prompt: &[u32],
        options: &Generation,
        cancellation: &Cancellation,
        emit: impl FnMut(u32) -> Result<()>,
    ) -> Result<Completion> {
        self.generate_with_prefill(prompt, options, Prefill::default(), cancellation, emit)
    }

    /// Generate with bounded, causal scheduling of uncached prompt blocks.
    ///
    /// The entire request, including all completed and partial blocks, shares
    /// one rollback checkpoint. Cancellation is checked between tokens inside
    /// each block. There is no intermediate commit or background work.
    ///
    /// # Errors
    /// Returns the errors of [`Self::generate`], and rejects an invalid prefill
    /// block size before capturing a checkpoint or changing decoder state.
    pub fn generate_with_prefill(
        &mut self,
        prompt: &[u32],
        options: &Generation,
        prefill: Prefill,
        cancellation: &Cancellation,
        emit: impl FnMut(u32) -> Result<()>,
    ) -> Result<Completion> {
        self.generate_checked(prompt, options, prefill, &|| cancellation.check(), emit)
    }

    /// Generate synchronously before an absolute monotonic request deadline.
    ///
    /// The deadline includes prefill, generation, callbacks and snapshot work.
    /// Expiry is cooperative: a running decoder operation or callback completes
    /// before the next check detects it. Recovery is allowed to finish after the
    /// deadline. Emitted output remains provisional until this call succeeds.
    /// Cancellation takes precedence when both cancellation and expiry occur.
    ///
    /// # Errors
    /// Returns an [`std::io::ErrorKind::TimedOut`] error on expiry, or the same
    /// errors as [`Self::generate`]. An expired request is rejected before any
    /// decoder operation. Admitted failures restore the previous committed
    /// state, unless restoration fails and closes admission until reset.
    pub fn generate_until(
        &mut self,
        prompt: &[u32],
        options: &Generation,
        cancellation: &Cancellation,
        deadline: Instant,
        emit: impl FnMut(u32) -> Result<()>,
    ) -> Result<Completion> {
        self.generate_until_with_prefill(
            prompt,
            options,
            Prefill::default(),
            cancellation,
            deadline,
            emit,
        )
    }

    /// Generate bounded causal prompt blocks before a monotonic deadline.
    ///
    /// Deadline and cancellation checks occur between tokens within blocks,
    /// with the same whole-request rollback and provisional output contract as
    /// [`Self::generate_until`]. Recovery may finish after the deadline.
    ///
    /// # Errors
    /// Returns the errors of [`Self::generate_until`] and rejects invalid
    /// prefill block sizes before any decoder operation.
    pub fn generate_until_with_prefill(
        &mut self,
        prompt: &[u32],
        options: &Generation,
        prefill: Prefill,
        cancellation: &Cancellation,
        deadline: Instant,
        emit: impl FnMut(u32) -> Result<()>,
    ) -> Result<Completion> {
        self.generate_checked(
            prompt,
            options,
            prefill,
            &|| check_deadline(cancellation, deadline, Instant::now()),
            emit,
        )
    }

    /// Generate with a caller-owned sampler and bounded causal prompt blocks.
    ///
    /// A successful request commits the sampler's state, including draws for a
    /// stop token. Any returned error restores its original state alongside the
    /// decoder, even if decoder restoration fails. The sampler's [`Clone`]
    /// implementation must independently snapshot all state changed by sampling;
    /// shared mutable RNG state cannot provide transactional recovery.
    /// External effects performed by a sampler cannot be rolled back.
    /// Cancellation is checked immediately before and after every sample.
    /// Emitted output remains provisional until the whole request succeeds.
    ///
    /// # Errors
    /// Returns the errors of [`Self::generate_with_prefill`], sampler failures,
    /// or an out-of-vocabulary sampled token.
    pub fn generate_sampled<S: Sampler + Clone>(
        &mut self,
        prompt: &[u32],
        options: &Generation,
        prefill: Prefill,
        cancellation: &Cancellation,
        sampler: &mut S,
        emit: impl FnMut(u32) -> Result<()>,
    ) -> Result<Completion> {
        self.generate_sampled_checked(
            prompt,
            options,
            prefill,
            &|| cancellation.check(),
            sampler,
            emit,
        )
    }

    /// Generate with a caller-owned sampler before a monotonic deadline.
    ///
    /// Uses the sampling and recovery contract of [`Self::generate_sampled`].
    /// Deadline checks surround sampling as well as decoder and callback work;
    /// a running operation completes before expiry is observed. Cancellation
    /// takes precedence over expiry, and recovery may finish after the deadline.
    ///
    /// # Errors
    /// Returns the errors of [`Self::generate_sampled`] or a timed-out error.
    #[allow(
        clippy::too_many_arguments,
        reason = "explicit sampling extends the existing bounded deadline API"
    )]
    pub fn generate_sampled_until<S: Sampler + Clone>(
        &mut self,
        prompt: &[u32],
        options: &Generation,
        prefill: Prefill,
        cancellation: &Cancellation,
        deadline: Instant,
        sampler: &mut S,
        emit: impl FnMut(u32) -> Result<()>,
    ) -> Result<Completion> {
        self.generate_sampled_checked(
            prompt,
            options,
            prefill,
            &|| check_deadline(cancellation, deadline, Instant::now()),
            sampler,
            emit,
        )
    }

    fn generate_checked(
        &mut self,
        prompt: &[u32],
        options: &Generation,
        prefill: Prefill,
        check: &impl Fn() -> Result<()>,
        emit: impl FnMut(u32) -> Result<()>,
    ) -> Result<Completion> {
        self.generate_sampled_checked(prompt, options, prefill, check, &mut Greedy, emit)
    }

    fn generate_sampled_checked<S: Sampler + Clone>(
        &mut self,
        prompt: &[u32],
        options: &Generation,
        prefill: Prefill,
        check: &impl Fn() -> Result<()>,
        sampler: &mut S,
        emit: impl FnMut(u32) -> Result<()>,
    ) -> Result<Completion> {
        let checkpoint = sampler.clone();
        let result = self.generate_transaction(prompt, options, prefill, check, sampler, emit);
        if result.is_err() {
            *sampler = checkpoint;
        }
        result
    }

    fn generate_transaction(
        &mut self,
        prompt: &[u32],
        options: &Generation,
        prefill: Prefill,
        check: &impl Fn() -> Result<()>,
        sampler: &mut impl Sampler,
        mut emit: impl FnMut(u32) -> Result<()>,
    ) -> Result<Completion> {
        if self.faulted {
            return Err(invalid(
                "native runtime requires reset after failed restoration",
            ));
        }
        check()?;
        self.validate_request(prompt, options, prefill)?;
        check()?;
        let rollback = self.capture(0, check)?;
        let result = (|| {
            check()?;
            let completion =
                self.generate_inner(prompt, options, prefill, check, sampler, &mut emit)?;
            check()?;
            // A failed optional capture still permits success, but deadline
            // expiry during capture must roll back the completed generation.
            let snapshot = if self.limits.checkpoints == 0 {
                None
            } else {
                let snapshot = self.capture(rollback.bytes, check).ok();
                check()?;
                snapshot
            };
            Ok((completion, snapshot))
        })();
        match result {
            Ok((completion, snapshot)) => {
                // Snapshots are an optional reuse optimization. A budget miss
                // does not invalidate an otherwise completed generation.
                if let Some(snapshot) = snapshot {
                    // Count pressure alone must not discard a usable prefix
                    // when its optional replacement could not be captured.
                    if self.checkpoints.len() == self.limits.checkpoints {
                        self.checkpoints.remove(0);
                    }
                    self.checkpoints.push(snapshot);
                }
                Ok(completion)
            }
            Err(error) => {
                self.faulted = true;
                let restored = self.decoder.restore(&rollback.state);
                // The public history always describes the committed request,
                // even if backend recovery fails and admission remains closed.
                self.tokens = rollback.tokens;
                self.logits = rollback.logits;
                restored?;
                self.faulted = false;
                Err(error)
            }
        }
    }

    fn validate_request(
        &self,
        prompt: &[u32],
        options: &Generation,
        prefill: Prefill,
    ) -> Result<()> {
        let length = prompt
            .len()
            .checked_add(options.max_tokens)
            .ok_or_else(|| invalid("native context length overflow"))?;
        if prompt.is_empty()
            || options.max_tokens == 0
            || prefill.block_tokens == 0
            || prefill.block_tokens > MAX_PREFILL_BLOCK_TOKENS
            || length > self.limits.context_tokens
            || options.stop_tokens.len() > 256
            || prompt.iter().chain(&options.stop_tokens).any(|&token| {
                usize::try_from(token).map_or(true, |value| value >= self.decoder.vocab_size())
            })
        {
            return Err(invalid(
                "invalid tokens or native context/generation/prefill bounds",
            ));
        }
        Ok(())
    }

    fn capture(
        &mut self,
        reserved: usize,
        check: &impl Fn() -> Result<()>,
    ) -> Result<Snapshot<D::Checkpoint>> {
        let logits_len = self.logits.len();
        let bytes = self
            .tokens
            .len()
            .checked_add(logits_len)
            .and_then(|length| length.checked_mul(4))
            .and_then(|size| size.checked_add(self.decoder.checkpoint_bytes()))
            .and_then(|size| size.checked_add(size_of::<Snapshot<D::Checkpoint>>()))
            .ok_or_else(|| invalid("native checkpoint byte count overflow"))?;
        if bytes > self.limits.checkpoint_bytes.saturating_sub(reserved) {
            return Err(invalid("native rollback checkpoint exceeds byte budget"));
        }
        while self
            .checkpoints
            .iter()
            .try_fold(reserved, |total, checkpoint| {
                total.checked_add(checkpoint.bytes)
            })
            .and_then(|size| size.checked_add(bytes))
            .is_none_or(|total| total > self.limits.checkpoint_bytes)
        {
            if self.checkpoints.is_empty() {
                return Err(invalid("native checkpoint budget exhausted"));
            }
            self.checkpoints.remove(0);
        }
        let tokens = copy(&self.tokens)?;
        let logits = copy(&self.logits)?;
        check()?;
        let state = self.decoder.checkpoint()?;
        Ok(Snapshot {
            state,
            tokens,
            logits,
            bytes,
        })
    }

    fn prepare(&mut self, prompt: &[u32], check: &impl Fn() -> Result<()>) -> Result<usize> {
        if !self.tokens.is_empty() && prompt.starts_with(&self.tokens) && !self.logits.is_empty() {
            return Ok(self.tokens.len());
        }
        let best = self
            .checkpoints
            .iter()
            .enumerate()
            .filter(|(_, snapshot)| prompt.starts_with(&snapshot.tokens))
            .max_by_key(|(_, snapshot)| snapshot.tokens.len())
            .map(|(index, _)| index);
        if let Some(index) = best {
            let snapshot = &self.checkpoints[index];
            self.tokens
                .try_reserve_exact(snapshot.tokens.len().saturating_sub(self.tokens.len()))?;
            self.logits.clear();
            self.logits.try_reserve_exact(snapshot.logits.len())?;
            check()?;
            self.decoder.restore(&snapshot.state)?;
            check()?;
            self.tokens.clear();
            self.tokens.extend_from_slice(&snapshot.tokens);
            self.logits.extend_from_slice(&snapshot.logits);
        } else {
            check()?;
            self.decoder.reset()?;
            check()?;
            self.tokens.clear();
            self.logits.clear();
        }
        Ok(self.tokens.len())
    }

    fn evaluate(&mut self, token: u32, check: &impl Fn() -> Result<()>) -> Result<()> {
        check()?;
        // Reserve before mutating decoder state; rollback is still available if
        // a decoder allocation or arithmetic operation subsequently fails.
        self.tokens.try_reserve(1)?;
        self.logits.clear();
        check()?;
        let result = self
            .decoder
            .evaluate_into(token, &mut self.logits)
            .and_then(|()| {
                if self.logits.len() != self.decoder.vocab_size()
                    || self.logits.iter().any(|x| !x.is_finite())
                {
                    return Err(invalid("native decoder returned invalid logits"));
                }
                check()
            });
        if let Err(error) = result {
            self.logits.clear();
            return Err(error);
        }
        self.tokens.push(token);
        Ok(())
    }

    fn ingest_block(&mut self, tokens: &[u32], check: &impl Fn() -> Result<()>) -> Result<()> {
        check()?;
        self.tokens.try_reserve(tokens.len())?;
        // Preserve the reusable allocation, but invalidate logits for the old
        // prefix before advancing state without producing a new distribution.
        self.logits.clear();
        check()?;
        self.decoder.ingest_block(tokens, check)?;
        check()?;
        self.tokens.extend_from_slice(tokens);
        Ok(())
    }

    fn generate_inner(
        &mut self,
        prompt: &[u32],
        options: &Generation,
        prefill: Prefill,
        check: &impl Fn() -> Result<()>,
        sampler: &mut impl Sampler,
        emit: &mut impl FnMut(u32) -> Result<()>,
    ) -> Result<Completion> {
        check()?;
        let reused = self.prepare(prompt, check)?;
        check()?;
        if let Some((&last, prefix)) = prompt[reused..].split_last() {
            for block in prefix.chunks(prefill.block_tokens) {
                self.ingest_block(block, check)?;
            }
            self.evaluate(last, check)?;
        }
        let mut completion = Completion {
            prompt_tokens: prompt.len(),
            reused_tokens: reused,
            generated_tokens: 0,
            finish: Finish::Length,
        };
        for _ in 0..options.max_tokens {
            check()?;
            let token = sampler.sample(&self.logits)?;
            check()?;
            if usize::try_from(token).map_or(true, |value| value >= self.decoder.vocab_size()) {
                return Err(invalid(
                    "native sampler returned an out-of-vocabulary token",
                ));
            }
            if options.stop_tokens.contains(&token) {
                completion.finish = Finish::Stop;
                break;
            }
            self.evaluate(token, check)?;
            check()?;
            emit(token)?;
            check()?;
            completion.generated_tokens += 1;
        }
        Ok(completion)
    }
}

fn check_deadline(cancellation: &Cancellation, deadline: Instant, now: Instant) -> Result<()> {
    cancellation.check()?;
    if now >= deadline {
        return Err(std::io::Error::new(
            std::io::ErrorKind::TimedOut,
            "native generation deadline exceeded",
        )
        .into());
    }
    Ok(())
}

fn copy<T: Copy>(source: &[T]) -> Result<Vec<T>> {
    let mut output = Vec::new();
    output.try_reserve_exact(source.len())?;
    output.extend_from_slice(source);
    Ok(output)
}

/// Select the greatest finite logit, using the lowest token ID to break ties.
///
/// # Errors
/// Rejects empty, oversized or nonfinite logits.
pub fn greedy(logits: &[f32]) -> Result<u32> {
    if logits.is_empty() || logits.len() > 1_048_576 {
        return Err(invalid("greedy sampling requires nonempty finite logits"));
    }
    let mut best = 0;
    let mut maximum = logits[0];
    for (index, &value) in logits.iter().enumerate() {
        // Validate every element even after finding an unbeatable maximum.
        // A strict comparison retains the lowest token ID for all ties,
        // including equal positive/negative zero values.
        if !value.is_finite() {
            return Err(invalid("greedy sampling requires nonempty finite logits"));
        }
        if value > maximum {
            best = index;
            maximum = value;
        }
    }
    Ok(u32::try_from(best)?)
}

#[cfg(test)]
mod deadline_tests {
    use super::*;
    use std::{
        cell::{Cell, RefCell},
        io::ErrorKind,
        rc::Rc,
        time::Duration,
    };

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum Operation {
        Reset,
        Restore,
        Checkpoint,
        Ingest(u32),
        Evaluate(u32),
        Sample,
        Emit(u32),
    }

    struct Clock {
        now: Cell<Instant>,
        deadline: Instant,
        expire_on: Cell<Option<Operation>>,
        operations: RefCell<Vec<Operation>>,
        restore_fail: Cell<bool>,
    }

    impl Clock {
        fn record(&self, operation: Operation) {
            self.operations.borrow_mut().push(operation);
            if self.expire_on.get() == Some(operation) {
                self.expire_on.set(None);
                self.now.set(self.deadline);
            }
        }

        fn check(&self) -> Result<()> {
            check_deadline(&Cancellation::default(), self.deadline, self.now.get())
        }
    }

    struct TimedDecoder {
        tokens: Vec<u32>,
        clock: Rc<Clock>,
    }

    impl Decoder for TimedDecoder {
        type Checkpoint = Vec<u32>;

        fn identity(&self) -> &'static str {
            "deadline-fixture/v1"
        }
        fn vocab_size(&self) -> usize {
            8
        }
        fn checkpoint_bytes(&self) -> usize {
            self.tokens.len() * 4
        }
        fn reset(&mut self) -> Result<()> {
            self.tokens.clear();
            self.clock.record(Operation::Reset);
            Ok(())
        }
        fn ingest(&mut self, token: u32) -> Result<()> {
            self.tokens.push(token);
            self.clock.record(Operation::Ingest(token));
            Ok(())
        }
        fn evaluate(&mut self, token: u32) -> Result<Vec<f32>> {
            self.tokens.push(token);
            self.clock.record(Operation::Evaluate(token));
            let mut logits = vec![-1.0; 8];
            logits[usize::try_from(self.tokens.iter().sum::<u32>() % 8)?] = 1.0;
            Ok(logits)
        }
        fn checkpoint(&self) -> Result<Self::Checkpoint> {
            self.clock.record(Operation::Checkpoint);
            Ok(self.tokens.clone())
        }
        fn restore(&mut self, checkpoint: &Self::Checkpoint) -> Result<()> {
            self.clock.record(Operation::Restore);
            if self.clock.restore_fail.get() {
                return Err(std::io::Error::other("injected deadline restoration failure").into());
            }
            self.tokens.clone_from(checkpoint);
            Ok(())
        }
    }

    const LIMITS: Limits = Limits {
        context_tokens: 32,
        checkpoints: 4,
        checkpoint_bytes: 16_384,
    };

    const OPTIONS: Generation = Generation {
        max_tokens: 1,
        stop_tokens: vec![],
    };

    #[derive(Clone)]
    struct TimedSampler {
        draws: usize,
        clock: Rc<Clock>,
    }

    impl Sampler for TimedSampler {
        fn sample(&mut self, logits: &[f32]) -> Result<u32> {
            self.draws += 1;
            self.clock.record(Operation::Sample);
            greedy(logits)
        }
    }

    fn seeded() -> Result<(Runtime<TimedDecoder>, Rc<Clock>)> {
        let now = Instant::now();
        let clock = Rc::new(Clock {
            now: Cell::new(now),
            deadline: now + Duration::from_secs(1),
            expire_on: Cell::new(None),
            operations: RefCell::new(Vec::new()),
            restore_fail: Cell::new(false),
        });
        let mut runtime = Runtime::new(
            TimedDecoder {
                tokens: Vec::new(),
                clock: clock.clone(),
            },
            LIMITS,
        )?;
        runtime.generate(&[1], &OPTIONS, &Cancellation::default(), |_| Ok(()))?;
        clock.operations.borrow_mut().clear();
        Ok((runtime, clock))
    }

    fn assert_timeout(result: Result<Completion>) -> Result<()> {
        let Err(error) = result else {
            return Err(invalid("deadline fixture unexpectedly completed"));
        };
        assert_eq!(
            error
                .downcast_ref::<std::io::Error>()
                .map(std::io::Error::kind),
            Some(ErrorKind::TimedOut)
        );
        Ok(())
    }

    fn assert_continues(runtime: &mut Runtime<TimedDecoder>) -> Result<()> {
        let committed = runtime.tokens().to_vec();
        let completion =
            runtime.generate(&committed, &OPTIONS, &Cancellation::default(), |_| Ok(()))?;
        assert_eq!(completion.reused_tokens, committed.len());
        let expected = committed.iter().sum::<u32>() % 8;
        assert_eq!(runtime.tokens().last(), Some(&expected));
        Ok(())
    }

    #[test]
    fn deadline_backend_expiry_rolls_back_prefill_and_generation() -> Result<()> {
        for operation in [
            Operation::Checkpoint,
            Operation::Reset,
            Operation::Ingest(2),
            Operation::Evaluate(3),
            Operation::Evaluate(5),
        ] {
            let (mut runtime, clock) = seeded()?;
            clock.expire_on.set(Some(operation));
            assert_timeout(runtime.generate_checked(
                &[2, 3],
                &OPTIONS,
                Prefill::default(),
                &|| clock.check(),
                |_| Err(invalid("expired decoder output must not emit")),
            ))?;
            assert!(clock.operations.borrow().contains(&operation));
            assert_eq!(clock.operations.borrow().last(), Some(&Operation::Restore));
            assert_eq!(runtime.tokens(), &[1, 1]);
            assert!(!runtime.is_faulted());
            assert_continues(&mut runtime)?;
        }
        Ok(())
    }

    #[test]
    fn deadline_expiry_during_prefix_restore_rolls_back() -> Result<()> {
        let (mut runtime, clock) = seeded()?;
        runtime.generate(&[1, 1, 2], &OPTIONS, &Cancellation::default(), |_| Ok(()))?;
        clock.operations.borrow_mut().clear();
        clock.expire_on.set(Some(Operation::Restore));
        assert_timeout(runtime.generate_checked(
            &[1, 1, 3],
            &OPTIONS,
            Prefill::default(),
            &|| clock.check(),
            |_| Err(invalid("expired prefix restore must not emit")),
        ))?;
        assert_eq!(
            *clock.operations.borrow(),
            [
                Operation::Checkpoint,
                Operation::Restore,
                Operation::Restore
            ]
        );
        assert_eq!(runtime.tokens(), &[1, 1, 2, 4]);
        assert!(!runtime.is_faulted());
        assert_continues(&mut runtime)
    }

    #[test]
    fn deadline_expiry_during_emit_or_optional_snapshot_rolls_back() -> Result<()> {
        for operation in [Operation::Emit(5), Operation::Checkpoint] {
            let (mut runtime, clock) = seeded()?;
            let mut emitted = Vec::new();
            assert_timeout(runtime.generate_checked(
                &[2, 3],
                &OPTIONS,
                Prefill::default(),
                &|| clock.check(),
                |token| {
                    emitted.push(token);
                    // Arm after the mandatory rollback capture so Checkpoint
                    // targets the optional completed-prefix capture exclusively.
                    clock.expire_on.set(Some(operation));
                    clock.record(Operation::Emit(token));
                    Ok(())
                },
            ))?;
            assert_eq!(emitted, [5]);
            assert_eq!(clock.operations.borrow().last(), Some(&Operation::Restore));
            assert_eq!(runtime.tokens(), &[1, 1]);
            assert_eq!(runtime.checkpoint_count(), 1);
            assert!(!runtime.is_faulted());
            assert_continues(&mut runtime)?;
        }
        Ok(())
    }

    #[test]
    fn deadline_restoration_failure_latches_admission_until_reset() -> Result<()> {
        let (mut runtime, clock) = seeded()?;
        clock.restore_fail.set(true);
        clock.expire_on.set(Some(Operation::Ingest(2)));
        let result = runtime.generate_checked(
            &[2, 3],
            &OPTIONS,
            Prefill::default(),
            &|| clock.check(),
            |_| Ok(()),
        );
        let Err(error) = result else {
            return Err(invalid(
                "failed deadline restoration unexpectedly completed",
            ));
        };
        assert!(
            error
                .to_string()
                .contains("injected deadline restoration failure")
        );
        assert!(runtime.is_faulted());
        assert_eq!(runtime.tokens(), &[1, 1]);
        clock.operations.borrow_mut().clear();
        assert!(
            runtime
                .generate(&[1, 1], &OPTIONS, &Cancellation::default(), |_| Ok(()))
                .is_err()
        );
        assert!(clock.operations.borrow().is_empty());
        clock.restore_fail.set(false);
        runtime.reset()?;
        runtime.generate(&[3], &OPTIONS, &Cancellation::default(), |_| Ok(()))?;
        assert_eq!(runtime.tokens(), &[3, 3]);
        assert!(!runtime.is_faulted());
        Ok(())
    }

    #[test]
    fn deadline_prefill_expiry_inside_blocks_restores_the_whole_request() -> Result<()> {
        for operation in [
            Operation::Ingest(3),
            Operation::Ingest(7),
            Operation::Evaluate(0),
        ] {
            let (mut runtime, clock) = seeded()?;
            clock.expire_on.set(Some(operation));
            assert_timeout(runtime.generate_checked(
                &[2, 3, 4, 5, 6, 7, 0],
                &OPTIONS,
                Prefill { block_tokens: 2 },
                &|| clock.check(),
                |_| Err(invalid("expired prefill must not emit")),
            ))?;
            assert!(clock.operations.borrow().contains(&operation));
            assert_eq!(clock.operations.borrow().last(), Some(&Operation::Restore));
            assert_eq!(runtime.tokens(), &[1, 1]);
            assert_eq!(runtime.decoder().tokens, [1, 1]);
            assert!(!runtime.is_faulted());
            assert_continues(&mut runtime)?;
        }
        Ok(())
    }

    #[test]
    fn sampled_deadline_during_sampling_delivery_or_snapshot_restores_caller_rng() -> Result<()> {
        for operation in [Operation::Sample, Operation::Emit(5), Operation::Checkpoint] {
            let (mut runtime, clock) = seeded()?;
            let initial_now = clock.now.get();
            let mut sampler = TimedSampler {
                draws: 7,
                clock: clock.clone(),
            };
            if operation == Operation::Sample {
                clock.expire_on.set(Some(operation));
            }
            let mut emitted = Vec::new();
            assert_timeout(runtime.generate_sampled_checked(
                &[2, 3],
                &OPTIONS,
                Prefill::default(),
                &|| clock.check(),
                &mut sampler,
                |token| {
                    emitted.push(token);
                    clock.expire_on.set(Some(operation));
                    clock.record(Operation::Emit(token));
                    Ok(())
                },
            ))?;
            assert_eq!(sampler.draws, 7);
            assert_eq!(runtime.tokens(), &[1, 1]);
            assert_eq!(runtime.decoder().tokens, [1, 1]);
            assert!(!runtime.is_faulted());
            assert_eq!(emitted.len(), usize::from(operation != Operation::Sample));
            clock.now.set(initial_now);
            clock.expire_on.set(None);
            let completion = runtime.generate_sampled_checked(
                &[2, 3],
                &OPTIONS,
                Prefill::default(),
                &|| clock.check(),
                &mut sampler,
                |token| {
                    assert_eq!(token, 5);
                    Ok(())
                },
            )?;
            assert_eq!(completion.generated_tokens, 1);
            assert_eq!(sampler.draws, 8);
        }
        Ok(())
    }

    #[test]
    fn sampled_deadline_after_stop_draw_or_failed_recovery_restores_caller_rng() -> Result<()> {
        for fail_restore in [false, true] {
            let (mut runtime, clock) = seeded()?;
            let mut sampler = TimedSampler {
                draws: 7,
                clock: clock.clone(),
            };
            clock.restore_fail.set(fail_restore);
            clock.expire_on.set(Some(Operation::Sample));
            assert!(
                runtime
                    .generate_sampled_checked(
                        &[2, 3],
                        &Generation {
                            max_tokens: 1,
                            stop_tokens: vec![5],
                        },
                        Prefill::default(),
                        &|| clock.check(),
                        &mut sampler,
                        |_| Err(invalid("expired stop sample must not emit")),
                    )
                    .is_err()
            );
            assert_eq!(sampler.draws, 7);
            assert_eq!(runtime.tokens(), &[1, 1]);
            assert_eq!(runtime.is_faulted(), fail_restore);
            assert!(!clock.operations.borrow().contains(&Operation::Evaluate(5)));
        }
        Ok(())
    }
}
