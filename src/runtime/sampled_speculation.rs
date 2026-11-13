//! Deterministic suffix proposals coupled to the caller's actual target draws.

use super::{
    Arc, Cancellation, Completion, Decoder, FiniteLogits, Finish, Generation, Instant, Prefill,
    Result, Runtime, Sampler, Snapshot, SpeculativeStats, SuffixDraftConfig, SuffixDrafter,
    MAX_SPECULATIVE_DRAFT_TOKENS, check_deadline, invalid,
};

struct SampledSuffixEffects<'a, S, E> {
    sampler: &'a mut S,
    emit: E,
}

struct SampledRound {
    emitted: usize,
    stopped: bool,
}

impl<D: Decoder> Runtime<D> {
    /// Generate sampled tokens with deterministic prompt/history suffix proposals.
    ///
    /// The caller's actual sampler draws once from each target distribution. A
    /// proposal is accepted exactly when that draw equals the proposed token;
    /// otherwise the draw is already a sample from the conditional residual of
    /// the point-mass proposal. This needs neither a second sampler nor additional
    /// acceptance draws. It preserves the caller's filtering, penalties and RNG
    /// order when the opted-in batch decoder produces identical per-position
    /// logits. This contract does not apply to probabilistic learned drafters.
    ///
    /// Eligible two-through-eight-token blocks use the decoder's explicit batched
    /// capability. Rejection or a stop inside a proposed block restores complete
    /// KV/recurrent/convolution state and replays only the accepted prefix.
    /// Simultaneous request and block checkpoints retain their full byte charge.
    /// Other decoders retain sequential target evaluation. No mode is selected
    /// automatically and this method establishes no measured speed benefit.
    ///
    /// Sampling, provisional callbacks, cancellation, deadlines, stop tokens and
    /// failed-request RNG recovery follow [`Self::generate_sampled_until`].
    /// A failed complete-state restore closes admission without retrying it.
    ///
    /// # Errors
    /// Returns the errors of [`Self::generate_sampled_until`], invalid suffix
    /// configuration, insufficient simultaneous checkpoint capacity or invalid
    /// batched decoder output.
    #[expect(
        clippy::too_many_arguments,
        reason = "Explicit suffix selection extends the established sampled deadline API."
    )]
    pub fn generate_sampled_speculative_until<S: Sampler + Clone>(
        &mut self,
        prompt: &[u32],
        options: &Generation,
        prefill: Prefill,
        suffix_config: SuffixDraftConfig,
        cancellation: &Cancellation,
        deadline: Instant,
        sampler: &mut S,
        emit: impl FnMut(u32) -> Result<()>,
    ) -> Result<(Completion, SpeculativeStats)> {
        self.generate_sampled_speculative_checked(
            prompt,
            options,
            prefill,
            suffix_config,
            &|| check_deadline(cancellation, deadline, Instant::now()),
            sampler,
            emit,
        )
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "The checked entry retains explicit suffix, sampler and callback ownership."
    )]
    fn generate_sampled_speculative_checked<S: Sampler + Clone>(
        &mut self,
        prompt: &[u32],
        options: &Generation,
        prefill: Prefill,
        suffix_config: SuffixDraftConfig,
        check: &impl Fn() -> Result<()>,
        sampler: &mut S,
        emit: impl FnMut(u32) -> Result<()>,
    ) -> Result<(Completion, SpeculativeStats)> {
        let sampler_rollback = sampler.clone();
        let result = self.sampled_suffix_transaction(
            prompt,
            options,
            prefill,
            suffix_config,
            check,
            &mut SampledSuffixEffects { sampler, emit },
        );
        if result.is_err() {
            *sampler = sampler_rollback;
        }
        result
    }

    fn sampled_suffix_transaction(
        &mut self,
        prompt: &[u32],
        options: &Generation,
        prefill: Prefill,
        suffix_config: SuffixDraftConfig,
        check: &impl Fn() -> Result<()>,
        effects: &mut SampledSuffixEffects<'_, impl Sampler, impl FnMut(u32) -> Result<()>>,
    ) -> Result<(Completion, SpeculativeStats)> {
        if self.faulted {
            return Err(invalid("native runtime requires reset after failed restoration"));
        }
        suffix_config.validate()?;
        check()?;
        self.validate_request(prompt, options, prefill)?;
        check()?;
        let rollback = self.capture(0, check)?;
        let result = (|| {
            let (completion, stats) = self.sampled_suffix_inner(
                prompt,
                options,
                prefill,
                suffix_config,
                check,
                effects,
                rollback.bytes,
            )?;
            check()?;
            let snapshot = if self.limits.checkpoints == 0 {
                None
            } else {
                let snapshot = self.capture(rollback.bytes, check).ok();
                check()?;
                snapshot
            };
            Ok((completion, stats, snapshot))
        })();
        match result {
            Ok((completion, stats, snapshot)) => {
                if let Some(snapshot) = snapshot {
                    if self.checkpoints.len() == self.limits.checkpoints {
                        self.checkpoints.remove(0);
                    }
                    self.checkpoints.push(snapshot);
                }
                Ok((completion, stats))
            }
            Err(error) => {
                // A failed round restore is terminal. In particular, do not
                // retry another restoration on the same provisional decoder.
                let already_faulted = self.faulted;
                let restored = if already_faulted {
                    Ok(())
                } else {
                    self.faulted = true;
                    self.decoder.restore(&rollback.state)
                };
                self.tokens = rollback.tokens;
                self.logits = rollback.logits;
                restored?;
                if !already_faulted {
                    self.faulted = false;
                }
                Err(error)
            }
        }
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "The request checkpoint reservation accompanies explicit generation effects."
    )]
    fn sampled_suffix_inner(
        &mut self,
        prompt: &[u32],
        options: &Generation,
        prefill: Prefill,
        suffix_config: SuffixDraftConfig,
        check: &impl Fn() -> Result<()>,
        effects: &mut SampledSuffixEffects<'_, impl Sampler, impl FnMut(u32) -> Result<()>>,
        reserved: usize,
    ) -> Result<(Completion, SpeculativeStats)> {
        check()?;
        let reused = self.prepare(prompt, check)?;
        check()?;
        let remaining = prompt.len() - reused;
        let capacity = self.decoder.batched_evaluation_capacity()
            .min(MAX_SPECULATIVE_DRAFT_TOKENS)
            .min(suffix_config.max_draft_tokens);
        // A rejection at position zero can waste the entire proposed tile
        // before one correction. Every further accepted token also bounds its
        // own replay. Reserve this worst case for the complete output cap,
        // including a possible stop draw, before starting prompt work.
        let factor = if capacity >= 2 { capacity + 1 } else { 1 };
        let evaluations = options.max_tokens.checked_mul(factor)
            .and_then(|tokens| tokens.checked_add(usize::from(remaining > 0)))
            .ok_or_else(|| invalid("sampled speculative forward work overflow"))?;
        self.decoder.admit_work(remaining.saturating_sub(1), evaluations)?;
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
        let mut stats = SpeculativeStats::default();
        let mut storage = [0; MAX_SPECULATIVE_DRAFT_TOKENS];
        while completion.generated_tokens < options.max_tokens {
            check()?;
            let remaining = options.max_tokens - completion.generated_tokens;
            let proposed = SuffixDrafter::propose_into(
                &self.tokens, suffix_config, remaining.min(capacity.max(1)), &mut storage,
            )?;
            if proposed == 0 {
                if self.sampled_suffix_single(options, check, effects, &mut stats)? {
                    completion.finish = Finish::Stop;
                    break;
                }
                completion.generated_tokens += 1;
                continue;
            }
            let round = self.sampled_suffix_round(
                &storage[..proposed], options, check, effects, &mut stats, reserved,
            )?;
            completion.generated_tokens += round.emitted;
            if round.stopped {
                completion.finish = Finish::Stop;
                break;
            }
            // The last accepted row is already the next target distribution.
            // Let the next suffix absorb that next draw into another shared
            // tile instead of forcing an ordinary one-token weight pass.
        }
        Ok((completion, stats))
    }

    fn sampled_suffix_token(
        &self,
        check: &impl Fn() -> Result<()>,
        sampler: &mut impl Sampler,
    ) -> Result<u32> {
        check()?;
        let token = sampler.sample_finite(FiniteLogits::from_validated(&self.logits)?)?;
        check()?;
        if !usize::try_from(token).is_ok_and(|value| value < self.decoder.vocab_size()) {
            return Err(invalid("native sampler returned an out-of-vocabulary token"));
        }
        Ok(token)
    }

    fn sampled_suffix_emit(
        token: u32,
        check: &impl Fn() -> Result<()>,
        emit: &mut impl FnMut(u32) -> Result<()>,
    ) -> Result<()> {
        check()?;
        emit(token)?;
        check()
    }

    fn sampled_suffix_single(
        &mut self,
        options: &Generation,
        check: &impl Fn() -> Result<()>,
        effects: &mut SampledSuffixEffects<'_, impl Sampler, impl FnMut(u32) -> Result<()>>,
        stats: &mut SpeculativeStats,
    ) -> Result<bool> {
        let token = self.sampled_suffix_token(check, effects.sampler)?;
        if options.stop_tokens.contains(&token) {
            return Ok(true);
        }
        self.evaluate(token, check)?;
        stats.target_evaluations += 1;
        Self::sampled_suffix_emit(token, check, &mut effects.emit)?;
        Ok(false)
    }

    fn sampled_suffix_round(
        &mut self,
        draft: &[u32],
        options: &Generation,
        check: &impl Fn() -> Result<()>,
        effects: &mut SampledSuffixEffects<'_, impl Sampler, impl FnMut(u32) -> Result<()>>,
        stats: &mut SpeculativeStats,
        reserved: usize,
    ) -> Result<SampledRound> {
        stats.draft_rounds += 1;
        stats.draft_tokens_proposed += draft.len();
        let mut batch = if draft.len() >= 2 && draft.len() <= self.decoder.batched_evaluation_capacity() {
            self.tokens.try_reserve(draft.len())?;
            let checkpoint = self.capture(reserved, check)?;
            let mut rows = Vec::new();
            check()?;
            self.decoder.evaluate_batch_into(draft, &mut rows, check)?;
            check()?;
            if rows.len() != draft.len() || rows.iter().any(|row| {
                row.len() != self.decoder.vocab_size()
                    || row.is_empty() || row.len() > 1_048_576
                    || !crate::kernels::is_finite_slice(row)
            }) {
                return Err(invalid("batched decoder returned invalid logits"));
            }
            stats.target_evaluations += draft.len();
            Some((checkpoint, rows))
        } else {
            None
        };
        for (index, &candidate) in draft.iter().enumerate() {
            let target = self.sampled_suffix_token(check, effects.sampler)?;
            let stopped = options.stop_tokens.contains(&target);
            if stopped || target != candidate {
                if let Some((checkpoint, _)) = batch {
                    self.sampled_suffix_restore(checkpoint, &draft[..index], check, stats)?;
                }
                if !stopped {
                    self.evaluate(target, check)?;
                    stats.target_evaluations += 1;
                    Self::sampled_suffix_emit(target, check, &mut effects.emit)?;
                }
                return Ok(SampledRound {
                    emitted: index + usize::from(!stopped), stopped,
                });
            }
            if let Some((_, rows)) = &mut batch {
                self.tokens.push(candidate);
                self.logits = Arc::new(std::mem::take(&mut rows[index]));
            } else {
                self.evaluate(candidate, check)?;
                stats.target_evaluations += 1;
            }
            stats.draft_tokens_accepted += 1;
            Self::sampled_suffix_emit(candidate, check, &mut effects.emit)?;
        }
        Ok(SampledRound { emitted: draft.len(), stopped: false })
    }

    fn sampled_suffix_restore(
        &mut self,
        checkpoint: Snapshot<D::Checkpoint>,
        accepted: &[u32],
        check: &impl Fn() -> Result<()>,
        stats: &mut SpeculativeStats,
    ) -> Result<()> {
        self.faulted = true;
        let restored = self.decoder.restore(&checkpoint.state);
        self.tokens = checkpoint.tokens;
        self.logits = checkpoint.logits;
        restored?;
        self.faulted = false;
        stats.rollback_restores += 1;
        check()?;
        for &token in accepted {
            self.evaluate(token, check)?;
            stats.target_evaluations += 1;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::Limits;
    use std::cell::Cell;
    use std::rc::Rc;
    use std::time::Duration;

    const PROMPT: [u32; 12] = [0, 1, 2, 0, 0, 1, 2, 0, 0, 1, 2, 0];
    const MATCHED: [u32; 6] = [0, 1, 2, 0, 0, 1];

    #[derive(Clone, Debug, Default, Eq, PartialEq)]
    struct HybridState {
        tokens: Vec<u32>,
        recurrent: u64,
        convolution: [u32; 3],
    }

    #[derive(Default)]
    struct BatchControl {
        state: HybridState,
        capacity: usize,
        batches: usize,
        restores: usize,
        malformed: bool,
        fail_restore: bool,
        fail_replay: bool,
        cancel_replay: Option<Cancellation>,
        admitted: Cell<Option<(usize, usize)>>,
    }

    impl Decoder for BatchControl {
        type Checkpoint = HybridState;

        fn identity(&self) -> &'static str { "sampled-suffix-hybrid-control" }
        fn vocab_size(&self) -> usize { 4 }
        fn checkpoint_bytes(&self) -> usize { 256 }
        fn admit_work(&self, ingest: usize, evaluate: usize) -> Result<()> {
            self.admitted.set(Some((ingest, evaluate)));
            Ok(())
        }
        fn reset(&mut self) -> Result<()> {
            self.state = HybridState::default();
            Ok(())
        }
        fn evaluate(&mut self, token: u32) -> Result<Vec<f32>> {
            if self.fail_replay && self.restores == 1 {
                return Err(invalid("injected accepted-prefix replay failure"));
            }
            if self.restores == 1 && let Some(cancellation) = &self.cancel_replay { cancellation.cancel(); }
            self.state.tokens.push(token);
            self.state.recurrent = self.state.recurrent.wrapping_mul(17).wrapping_add(u64::from(token) + 1);
            self.state.convolution.rotate_left(1);
            self.state.convolution[2] = token;
            let value = f32::from(u8::try_from(self.state.recurrent & 255)?);
            Ok(vec![value, value * 0.5, -value, value * -0.5])
        }
        fn batched_evaluation_capacity(&self) -> usize { self.capacity }
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
            if self.malformed {
                let last = rows.len() - 1;
                rows[last][0] = f32::NAN;
            }
            Ok(())
        }
        fn checkpoint(&self) -> Result<Self::Checkpoint> { Ok(self.state.clone()) }
        fn restore(&mut self, checkpoint: &Self::Checkpoint) -> Result<()> {
            self.restores += 1;
            if self.fail_restore {
                return Err(invalid("injected complete-state restore failure"));
            }
            self.state = checkpoint.clone();
            Ok(())
        }
    }

    #[derive(Clone, Default)]
    struct ScriptSampler {
        script: Vec<u32>,
        draws: usize,
        observed: Vec<Vec<u32>>,
        signal: Option<Rc<Cell<usize>>>,
        cancel: Option<Cancellation>,
    }

    impl Sampler for ScriptSampler {
        fn sample(&mut self, logits: &[f32]) -> Result<u32> {
            self.observed.push(logits.iter().map(|value| value.to_bits()).collect());
            let token = self.script[self.draws];
            self.draws += 1;
            if let Some(signal) = &self.signal { signal.set(self.draws); }
            if let Some(cancellation) = &self.cancel { cancellation.cancel(); }
            Ok(token)
        }
    }

    fn runtime(capacity: usize) -> Result<Runtime<BatchControl>> {
        Runtime::new(BatchControl { capacity, ..BatchControl::default() }, Limits {
            context_tokens: 64, checkpoints: 0, checkpoint_bytes: 65_536,
        })
    }

    fn suffix() -> Result<SuffixDraftConfig> {
        let config = SuffixDraftConfig::new(1, 8, 4)?;
        // Repeated single tokens select a more recent one-token overlapping
        // continuation. This period-four history really proposes four tokens.
        assert_eq!(SuffixDrafter::propose(&PROMPT, config, 4)?, MATCHED[..4]);
        Ok(config)
    }

    #[test]
    fn sampled_suffix_matches_target_draws_logits_and_full_state_at_every_rejection() -> Result<()> {
        for capacity in [0, 8] {
            for rejected in 0..=4 {
                let mut ordinary = runtime(capacity)?;
                let mut speculative = runtime(capacity)?;
                let mut script = (0..16).map(|index| MATCHED[index % 4]).collect::<Vec<_>>();
                if rejected < 4 { script[rejected] = 3; }
                let mut expected_sampler = ScriptSampler { script, ..ScriptSampler::default() };
                let mut actual_sampler = expected_sampler.clone();
                let mut expected = Vec::new();
                let mut actual = Vec::new();
                let options = Generation { max_tokens: 16, stop_tokens: Vec::new() };
                let deadline = Instant::now() + Duration::from_secs(60);
                let expected_completion = ordinary.generate_sampled_until(
                    &PROMPT, &options, Prefill::default(), &Cancellation::default(), deadline,
                    &mut expected_sampler, |token| { expected.push(token); Ok(()) },
                )?;
                let (actual_completion, stats) = speculative.generate_sampled_speculative_until(
                    &PROMPT, &options, Prefill::default(), suffix()?, &Cancellation::default(), deadline,
                    &mut actual_sampler, |token| { actual.push(token); Ok(()) },
                )?;
                assert_eq!(actual_completion, expected_completion);
                assert_eq!(actual, expected);
                assert_eq!(actual_sampler.draws, expected_sampler.draws);
                assert_eq!(actual_sampler.observed, expected_sampler.observed);
                assert_eq!(speculative.tokens, ordinary.tokens);
                assert_eq!(speculative.logits, ordinary.logits);
                assert_eq!(speculative.decoder.state, ordinary.decoder.state);
                assert!(stats.target_evaluations >= actual.len());
                assert_eq!(speculative.decoder.batches > 0, capacity > 0);
                if capacity > 0 && rejected == 4 {
                    assert_eq!(speculative.decoder.batches, 4);
                    assert_eq!(stats.target_evaluations, 16);
                    assert_eq!(stats.rollback_restores, 0);
                }
                if capacity > 0 && rejected < 4 {
                    assert!(stats.rollback_restores > 0);
                }
            }
        }
        Ok(())
    }

    #[test]
    fn sampled_suffix_stop_inside_each_tile_excludes_stop_from_complete_state() -> Result<()> {
        for stop_at in 0..=4 {
            let mut ordinary = runtime(8)?;
            let mut speculative = runtime(8)?;
            let mut script = MATCHED.to_vec();
            script[stop_at] = 3;
            let mut expected_sampler = ScriptSampler { script, ..ScriptSampler::default() };
            let mut actual_sampler = expected_sampler.clone();
            let mut expected = Vec::new();
            let mut actual = Vec::new();
            let options = Generation { max_tokens: 6, stop_tokens: vec![3] };
            let deadline = Instant::now() + Duration::from_secs(60);
            let expected_completion = ordinary.generate_sampled_until(
                &PROMPT, &options, Prefill::default(), &Cancellation::default(), deadline,
                &mut expected_sampler, |token| { expected.push(token); Ok(()) },
            )?;
            let (actual_completion, _) = speculative.generate_sampled_speculative_until(
                &PROMPT, &options, Prefill::default(), suffix()?, &Cancellation::default(), deadline,
                &mut actual_sampler, |token| { actual.push(token); Ok(()) },
            )?;
            assert_eq!(actual_completion, expected_completion);
            assert_eq!(actual_completion.finish, Finish::Stop);
            assert_eq!(actual_completion.generated_tokens, stop_at);
            assert_eq!(actual, expected);
            assert_eq!(actual_sampler.draws, stop_at + 1);
            assert_eq!(actual_sampler.observed, expected_sampler.observed);
            assert_eq!(speculative.tokens, ordinary.tokens);
            assert_eq!(speculative.logits, ordinary.logits);
            assert_eq!(speculative.decoder.state, ordinary.decoder.state);
        }
        Ok(())
    }

    #[test]
    fn sampled_suffix_matching_stop_restores_the_entire_proposed_suffix() -> Result<()> {
        // A stop can equal the proposed token itself. The evaluated stop and
        // every following proposed state still have to be removed.
        let mut speculative = runtime(8)?;
        let mut sampler = ScriptSampler { script: vec![0], ..ScriptSampler::default() };
        let (completion, stats) = speculative.generate_sampled_speculative_until(
            &PROMPT, &Generation { max_tokens: 4, stop_tokens: vec![0] }, Prefill::default(),
            suffix()?, &Cancellation::default(), Instant::now() + Duration::from_secs(60),
            &mut sampler, |_| Err(invalid("a sampled stop must never be delivered")),
        )?;
        assert_eq!(completion.generated_tokens, 0);
        assert_eq!(completion.finish, Finish::Stop);
        assert_eq!(speculative.tokens, PROMPT);
        assert_eq!(speculative.decoder.state.tokens, PROMPT);
        assert_eq!(stats.draft_tokens_accepted, 0);
        assert_eq!(stats.rollback_restores, 1);
        assert_eq!(sampler.draws, 1);
        Ok(())
    }

    #[test]
    fn sampled_suffix_preserves_seeded_reference_sampler_continuation() -> Result<()> {
        use crate::sampling::{ReferenceSampler, Sampling};

        for seed in [0, 17, 99, 999] {
            let mut ordinary = runtime(8)?;
            let mut speculative = runtime(8)?;
            let mut expected_sampler = ReferenceSampler::new(Sampling {
                temperature: 80.0, top_k: 4, top_p: 1.0,
            }, seed)?;
            let mut actual_sampler = expected_sampler.clone();
            let mut expected = Vec::new();
            let mut actual = Vec::new();
            let options = Generation { max_tokens: 6, stop_tokens: Vec::new() };
            let deadline = Instant::now() + Duration::from_secs(60);
            let expected_completion = ordinary.generate_sampled_until(
                &PROMPT, &options, Prefill::default(), &Cancellation::default(), deadline,
                &mut expected_sampler, |token| { expected.push(token); Ok(()) },
            )?;
            let (actual_completion, _) = speculative.generate_sampled_speculative_until(
                &PROMPT, &options, Prefill::default(), suffix()?, &Cancellation::default(), deadline,
                &mut actual_sampler, |token| { actual.push(token); Ok(()) },
            )?;
            assert_eq!(actual, expected);
            assert_eq!(actual_completion, expected_completion);
            assert_eq!(speculative.decoder.state, ordinary.decoder.state);
            assert_eq!(speculative.logits, ordinary.logits);
            for _ in 0..10 {
                assert_eq!(actual_sampler.sample(&[0.0; 4])?, expected_sampler.sample(&[0.0; 4])?);
            }
        }
        Ok(())
    }

    #[test]
    fn sampled_suffix_failures_restore_caller_rng_and_request_or_latch_failed_restore() -> Result<()> {
        for failure in 0..7 {
            let mut speculative = runtime(8)?;
            let cancellation = Cancellation::default();
            let mut sampler = ScriptSampler {
                script: vec![0, 3, 2, 0],
                cancel: (failure == 2).then(|| cancellation.clone()),
                ..ScriptSampler::default()
            };
            speculative.decoder.malformed = failure == 3;
            speculative.decoder.fail_restore = failure == 4;
            speculative.decoder.fail_replay = failure == 5;
            speculative.decoder.cancel_replay = (failure == 6).then(|| cancellation.clone());
            let options = Generation { max_tokens: 4, stop_tokens: Vec::new() };
            let mut delivered = 0;
            let result = speculative.generate_sampled_speculative_until(
                &PROMPT, &options, Prefill::default(), suffix()?, &cancellation,
                Instant::now() + Duration::from_secs(60), &mut sampler,
                |_| {
                    delivered += 1;
                    if failure == 0 { return Err(invalid("injected callback failure")); }
                    if failure == 1 { cancellation.cancel(); }
                    Ok(())
                },
            );
            assert!(result.is_err());
            assert_eq!(sampler.draws, 0);
            assert!(sampler.observed.is_empty());
            assert!(speculative.tokens.is_empty());
            assert!(speculative.logits.is_empty());
            assert_eq!(speculative.is_faulted(), failure == 4);
            if failure == 4 {
                assert_eq!(speculative.decoder.restores, 1);
            } else {
                assert_eq!(speculative.decoder.state, HybridState::default());
            }
            if failure == 3 { assert_eq!(delivered, 0); }
            if failure == 5 || failure == 6 { assert_eq!(speculative.decoder.restores, 2); }
            if failure != 4 {
                speculative.decoder.malformed = false;
                speculative.decoder.fail_replay = false;
                speculative.decoder.cancel_replay = None;
                sampler.cancel = None;
                let _ = speculative.generate_sampled_speculative_until(
                    &PROMPT, &options, Prefill::default(), suffix()?, &Cancellation::default(),
                    Instant::now() + Duration::from_secs(60), &mut sampler, |_| Ok(()),
                )?;
                assert_eq!(sampler.draws, 4);
                assert!(!speculative.is_faulted());
            }
        }
        Ok(())
    }

    #[test]
    fn sampled_suffix_charges_both_snapshots_before_batch_and_all_rejected_work() -> Result<()> {
        let mut speculative = runtime(8)?;
        let request = speculative.capture(0, &|| Ok(()))?;
        speculative.limits.checkpoint_bytes = request.bytes + 1;
        drop(request);
        let mut sampler = ScriptSampler { script: vec![0; 4], ..ScriptSampler::default() };
        let mut delivered = 0;
        let result = speculative.generate_sampled_speculative_until(
            &PROMPT, &Generation { max_tokens: 4, stop_tokens: Vec::new() }, Prefill::default(),
            suffix()?, &Cancellation::default(), Instant::now() + Duration::from_secs(60),
            &mut sampler, |_| { delivered += 1; Ok(()) },
        );
        assert!(result.is_err());
        assert_eq!(speculative.decoder.batches, 0);
        assert_eq!(speculative.decoder.admitted.get(), Some((11, 21)));
        assert_eq!(sampler.draws, 0);
        assert_eq!(delivered, 0);
        assert_eq!(speculative.decoder.state, HybridState::default());
        assert!(!speculative.is_faulted());
        Ok(())
    }

    #[test]
    fn sampled_suffix_deadline_after_draw_restores_rng_before_any_callback() -> Result<()> {
        let mut speculative = runtime(8)?;
        let draws = Rc::new(Cell::new(0));
        let mut sampler = ScriptSampler {
            script: vec![0; 4], signal: Some(draws.clone()), ..ScriptSampler::default()
        };
        let mut delivered = 0;
        let result = speculative.generate_sampled_speculative_checked(
            &PROMPT, &Generation { max_tokens: 4, stop_tokens: Vec::new() }, Prefill::default(), suffix()?,
            &|| if draws.get() == 0 { Ok(()) } else {
                Err(std::io::Error::new(std::io::ErrorKind::TimedOut, "injected post-draw deadline").into())
            },
            &mut sampler, |_| { delivered += 1; Ok(()) },
        );
        assert!(result.is_err());
        assert_eq!(sampler.draws, 0);
        assert_eq!(delivered, 0);
        assert_eq!(speculative.decoder.state, HybridState::default());
        assert!(!speculative.is_faulted());
        Ok(())
    }
}
