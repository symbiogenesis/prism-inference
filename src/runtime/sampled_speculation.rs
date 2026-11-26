//! Deterministic suffix proposals coupled to the caller's actual target draws.

use super::{
    Arc, Cancellation, Completion, Decoder, Finish, FiniteLogits, Generation, Instant,
    MAX_SPECULATIVE_DRAFT_TOKENS, Prefill, PromptEndCapture, Result, Runtime, Sampler, Snapshot,
    SpeculativeStats, SuffixDraftConfig, VerifiedRound, check_deadline, invalid,
};
use crate::speculative::{
    AdaptiveGateConfig, AdaptiveGateDecision, AdaptiveSpeculationGate, MovingSourceCursor,
};

struct SampledSuffixEffects<'a, S, E> {
    sampler: &'a mut S,
    emit: E,
}

struct SampledRound {
    emitted: usize,
    stopped: bool,
}

type PreparedSuffixBatch<C> = Option<(Snapshot<C>, Vec<Vec<f32>>)>;

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
        deadline: impl Into<Option<Instant>>,
        sampler: &mut S,
        emit: impl FnMut(u32) -> Result<()>,
    ) -> Result<(Completion, SpeculativeStats)> {
        self.generate_sampled_speculative_with_gate_until(
            prompt,
            options,
            prefill,
            suffix_config,
            AdaptiveGateConfig::default(),
            cancellation,
            deadline,
            sampler,
            emit,
        )
    }

    /// Generate sampled tokens with bounded moving-cursor suffix proposals and an explicit
    /// [`AdaptiveGateConfig`].
    ///
    /// # Errors
    /// Returns the errors of [`Self::generate_sampled_speculative_until`] or an invalid
    /// [`AdaptiveGateConfig`].
    #[expect(
        clippy::too_many_arguments,
        reason = "Explicit adaptive gate selection extends the established sampled speculation API."
    )]
    pub fn generate_sampled_speculative_with_gate_until<S: Sampler + Clone>(
        &mut self,
        prompt: &[u32],
        options: &Generation,
        prefill: Prefill,
        suffix_config: SuffixDraftConfig,
        gate_config: AdaptiveGateConfig,
        cancellation: &Cancellation,
        deadline: impl Into<Option<Instant>>,
        sampler: &mut S,
        emit: impl FnMut(u32) -> Result<()>,
    ) -> Result<(Completion, SpeculativeStats)> {
        let deadline = deadline.into();
        self.generate_sampled_speculative_checked(
            prompt,
            options,
            prefill,
            suffix_config,
            gate_config,
            &|| check_deadline(cancellation, deadline, Instant::now()),
            sampler,
            emit,
        )
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "The checked entry retains explicit suffix, gate, sampler and callback ownership."
    )]
    fn generate_sampled_speculative_checked<S: Sampler + Clone>(
        &mut self,
        prompt: &[u32],
        options: &Generation,
        prefill: Prefill,
        suffix_config: SuffixDraftConfig,
        gate_config: AdaptiveGateConfig,
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
            gate_config,
            check,
            &mut SampledSuffixEffects { sampler, emit },
        );
        if result.is_err() {
            *sampler = sampler_rollback;
        }
        result
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "Keep transactional rollback explicit around suffix, gate, and folded decoder state."
    )]
    fn sampled_suffix_transaction(
        &mut self,
        prompt: &[u32],
        options: &Generation,
        prefill: Prefill,
        suffix_config: SuffixDraftConfig,
        gate_config: AdaptiveGateConfig,
        check: &impl Fn() -> Result<()>,
        effects: &mut SampledSuffixEffects<'_, impl Sampler, impl FnMut(u32) -> Result<()>>,
    ) -> Result<(Completion, SpeculativeStats)> {
        if self.faulted {
            return Err(invalid(
                "native runtime requires reset after failed restoration",
            ));
        }
        suffix_config.validate()?;
        gate_config.validate()?;
        check()?;
        self.validate_request(prompt, options, prefill)?;
        check()?;
        let rollback = self.capture_transient(0, check)?;
        let mut prompt_end = None;
        let result = (|| {
            let mut prompt_end_capture = PromptEndCapture {
                reserved: rollback.bytes,
                slot: &mut prompt_end,
            };
            let (completion, stats) = self.sampled_suffix_inner(
                prompt,
                options,
                prefill,
                suffix_config,
                gate_config,
                check,
                effects,
                &mut prompt_end_capture,
            )?;
            check()?;
            let snapshot = if self.limits.checkpoints == 0 {
                None
            } else {
                let snapshot_reserved = rollback
                    .bytes
                    .saturating_add(prompt_end.as_ref().map_or(0, |s| s.bytes));
                let snapshot = self.capture_reusable(snapshot_reserved, check).ok();
                check()?;
                snapshot
            };
            Ok((completion, stats, snapshot))
        })();
        match result {
            Ok((completion, stats, snapshot)) => {
                self.commit_prompt_end(prompt_end);
                self.commit_reusable_snapshot(snapshot);
                self.recycle_snapshot(rollback);
                Ok((completion, stats))
            }
            Err(error) => {
                // A failed round restore is terminal. In particular, do not
                // retry another restoration on the same provisional decoder.
                self.decoder.discard_folded_verification();
                let Snapshot {
                    state,
                    tokens,
                    media_spans,
                    logits,
                    ..
                } = rollback;
                let already_faulted = self.faulted;
                let restored = if already_faulted {
                    Ok(())
                } else {
                    self.faulted = true;
                    self.decoder.restore(&state)
                };
                self.tokens = tokens;
                self.media_spans = media_spans;
                let previous_logits = std::mem::replace(&mut self.logits, logits);
                restored?;
                if !already_faulted {
                    self.faulted = false;
                }
                self.recycle_checkpoint(state);
                self.recycle_logits(previous_logits);
                self.recycle_optional_snapshot(prompt_end);
                Err(error)
            }
        }
    }

    #[expect(
        clippy::too_many_arguments,
        clippy::too_many_lines,
        reason = "The request checkpoint reservation accompanies explicit generation effects."
    )]
    fn sampled_suffix_inner(
        &mut self,
        prompt: &[u32],
        options: &Generation,
        prefill: Prefill,
        suffix_config: SuffixDraftConfig,
        gate_config: AdaptiveGateConfig,
        check: &impl Fn() -> Result<()>,
        effects: &mut SampledSuffixEffects<'_, impl Sampler, impl FnMut(u32) -> Result<()>>,
        prompt_end: &mut PromptEndCapture<'_, D::Checkpoint>,
    ) -> Result<(Completion, SpeculativeStats)> {
        check()?;
        let reused = self.prepare(prompt, check)?;
        check()?;
        let remaining = prompt.len() - reused;
        let capacity = self
            .decoder
            .batched_evaluation_capacity()
            .min(MAX_SPECULATIVE_DRAFT_TOKENS)
            .min(suffix_config.max_draft_tokens);
        // A rejection at position zero can waste the entire proposed tile
        // before one correction. Every further accepted token also bounds its
        // own replay. Reserve this worst case for the complete output cap,
        // including a possible stop draw, before starting prompt work.
        let factor = if capacity >= 2 { capacity + 1 } else { 1 };
        let evaluations = options
            .max_tokens
            .checked_mul(factor)
            .and_then(|tokens| tokens.checked_add(usize::from(remaining > 0)))
            .ok_or_else(|| invalid("sampled speculative forward work overflow"))?;
        self.decoder
            .admit_work(remaining.saturating_sub(1), evaluations)?;
        check()?;
        if let Some((&last, prefix)) = prompt[reused..].split_last() {
            for block in prefix.chunks(prefill.block_tokens) {
                self.ingest_block(block, check)?;
            }
            self.evaluate(last, check)?;
        }
        self.maybe_capture_prompt_end(prompt, prompt_end, check)?;
        let reserved = prompt_end
            .reserved
            .saturating_add(prompt_end.slot.as_ref().map_or(0, |s| s.bytes));
        let mut completion = Completion {
            prompt_tokens: prompt.len(),
            reused_tokens: reused,
            generated_tokens: 0,
            finish: Finish::Length,
        };
        let mut stats = SpeculativeStats::default();
        let mut storage = [0; MAX_SPECULATIVE_DRAFT_TOKENS];
        let mut cursor = MovingSourceCursor::new();
        let mut gate = AdaptiveSpeculationGate::new(gate_config)?;
        let folded_enabled = self.decoder.supports_folded_verification() && capacity >= 2;
        let mut pending_anchor: Option<u32> = None;
        let mut carried_tokens = Vec::with_capacity(MAX_SPECULATIVE_DRAFT_TOKENS);

        while completion.generated_tokens < options.max_tokens {
            check()?;
            let remaining = options.max_tokens - completion.generated_tokens;

            if capacity >= 2 && gate.decide() == AdaptiveGateDecision::SerialBurst {
                self.flush_folded_and_anchor(
                    &mut pending_anchor,
                    &mut carried_tokens,
                    check,
                    &mut stats,
                    &mut gate,
                )?;
                let serial_start = Instant::now();
                let stopped = self.sampled_suffix_single(options, check, effects, &mut stats)?;
                gate.record_serial_step(serial_start.elapsed());
                stats.record_adaptive_serial_burst(true);
                cursor.observe_round(&self.tokens, 0, 0, false);
                if stopped {
                    completion.finish = Finish::Stop;
                    break;
                }
                completion.generated_tokens += 1;
                continue;
            }

            self.suffix_tree.sync(&self.tokens)?;
            let anchor_len = usize::from(pending_anchor.is_some());
            let draft_limit = if folded_enabled {
                remaining.min(capacity.saturating_sub(anchor_len))
            } else {
                remaining.min(capacity.max(1))
            };
            let cursor_draft = if draft_limit > 0 {
                cursor.propose_with_tree(
                    &self.suffix_tree,
                    suffix_config,
                    draft_limit,
                    &mut storage,
                )?
            } else {
                crate::speculative::CursorDraft::default()
            };
            if cursor_draft.cursor_tokens > 0 {
                stats.record_cursor_draft(cursor_draft.cursor_tokens);
            }

            let block_len = anchor_len + cursor_draft.len;
            if folded_enabled && cursor_draft.len > 0 && block_len >= 2 {
                let mut block = [0_u32; MAX_SPECULATIVE_DRAFT_TOKENS];
                if let Some(anchor) = pending_anchor.take() {
                    block[0] = anchor;
                }
                block[anchor_len..block_len].copy_from_slice(&storage[..cursor_draft.len]);
                let (round, accepted_draft) = self.sampled_folded_round(
                    &mut carried_tokens,
                    &mut pending_anchor,
                    &block[..block_len],
                    anchor_len,
                    options,
                    check,
                    effects,
                    &mut stats,
                    &mut gate,
                )?;
                cursor.observe_round(&self.tokens, cursor_draft.len, accepted_draft, false);
                completion.generated_tokens += round.emitted;
                if round.stopped {
                    self.flush_folded_and_anchor(
                        &mut pending_anchor,
                        &mut carried_tokens,
                        check,
                        &mut stats,
                        &mut gate,
                    )?;
                    completion.finish = Finish::Stop;
                    break;
                }
                continue;
            }

            self.flush_folded_and_anchor(
                &mut pending_anchor,
                &mut carried_tokens,
                check,
                &mut stats,
                &mut gate,
            )?;

            if cursor_draft.len == 0 {
                let serial_start = Instant::now();
                let stopped = self.sampled_suffix_single(options, check, effects, &mut stats)?;
                gate.record_serial_step(serial_start.elapsed());
                cursor.observe_round(&self.tokens, 0, 0, false);
                if stopped {
                    completion.finish = Finish::Stop;
                    break;
                }
                completion.generated_tokens += 1;
                continue;
            }

            let verify_start = Instant::now();
            let draft_slice = &storage[..cursor_draft.len];
            let (round, accepted_draft) = self.sampled_suffix_round_with_outcome(
                draft_slice,
                options,
                check,
                effects,
                &mut stats,
                reserved,
            )?;
            gate.record_verify_round(
                cursor_draft.len,
                accepted_draft,
                round.emitted,
                verify_start.elapsed(),
                std::time::Duration::ZERO,
            );
            cursor.observe_round(&self.tokens, cursor_draft.len, accepted_draft, false);
            completion.generated_tokens += round.emitted;
            if round.stopped {
                completion.finish = Finish::Stop;
                break;
            }
            // The last accepted row is already the next target distribution.
            // Let the next suffix absorb that next draw into another shared
            // tile instead of forcing an ordinary one-token weight pass.
        }
        self.flush_folded_and_anchor(
            &mut pending_anchor,
            &mut carried_tokens,
            check,
            &mut stats,
            &mut gate,
        )?;
        Ok((completion, stats))
    }

    fn evaluate_committed_anchor(
        &mut self,
        token: u32,
        check: &impl Fn() -> Result<()>,
    ) -> Result<()> {
        check()?;
        let mut rows = self.take_batch_rows(1)?;
        let Some(mut row) = rows.pop() else {
            return Err(invalid("missing batch logit row for committed anchor"));
        };
        row.clear();
        check()?;
        let res = self
            .decoder
            .evaluate_finite_into(token, &mut row)
            .map(|logits| (logits.as_slice().as_ptr(), logits.as_slice().len()));
        match res {
            Ok((pointer, length)) => {
                if length != self.decoder.vocab_size()
                    || length != row.len()
                    || !std::ptr::eq(pointer, row.as_ptr())
                {
                    rows.push(row);
                    self.recycle_batch_rows(rows);
                    return Err(invalid("native decoder returned invalid anchor logits"));
                }
                check()?;
                let prev = std::mem::replace(&mut self.logits, Arc::new(row));
                self.recycle_logits(prev);
                self.recycle_batch_rows(rows);
                Ok(())
            }
            Err(error) => {
                rows.push(row);
                self.recycle_batch_rows(rows);
                Err(error)
            }
        }
    }

    fn flush_folded_and_anchor(
        &mut self,
        pending_anchor: &mut Option<u32>,
        carried_tokens: &mut Vec<u32>,
        check: &impl Fn() -> Result<()>,
        stats: &mut SpeculativeStats,
        gate: &mut AdaptiveSpeculationGate,
    ) -> Result<()> {
        if self.decoder.has_pending_folded_verification() {
            let flush_start = Instant::now();
            self.decoder.flush_folded_verification(check)?;
            let elapsed = flush_start.elapsed();
            if !carried_tokens.is_empty() {
                stats.record_folded_flush();
                gate.charge_flush_overhead(elapsed);
            }
        }
        carried_tokens.clear();
        if let Some(anchor) = pending_anchor.take() {
            let serial_start = Instant::now();
            self.evaluate_committed_anchor(anchor, check)?;
            gate.record_serial_step(serial_start.elapsed());
        }
        Ok(())
    }

    #[expect(
        clippy::too_many_arguments,
        clippy::too_many_lines,
        reason = "Folded verification round preserves exact token, sampler, carry, and gate invariants."
    )]
    fn sampled_folded_round(
        &mut self,
        carried_tokens: &mut Vec<u32>,
        pending_anchor: &mut Option<u32>,
        block: &[u32],
        anchor_len: usize,
        options: &Generation,
        check: &impl Fn() -> Result<()>,
        effects: &mut SampledSuffixEffects<'_, impl Sampler, impl FnMut(u32) -> Result<()>>,
        stats: &mut SpeculativeStats,
        gate: &mut AdaptiveSpeculationGate,
    ) -> Result<(SampledRound, usize)> {
        let verify_start = Instant::now();
        let draft = &block[anchor_len..];
        let cap_before = self.tokens.capacity();
        self.tokens.try_reserve(block.len())?;
        if self.tokens.capacity() > cap_before {
            self.reallocation_witness.growth_reallocations = self
                .reallocation_witness
                .growth_reallocations
                .saturating_add(1);
        }
        let mut rows = self.take_batch_rows(block.len())?;
        let mut row_caps_before = [0_usize; MAX_SPECULATIVE_DRAFT_TOKENS];
        for (dst, row) in row_caps_before.iter_mut().zip(&rows) {
            *dst = row.capacity();
        }
        check()?;
        let eval_res = self.decoder.evaluate_verification_block_folded(
            carried_tokens,
            block,
            &mut rows,
            check,
        );
        if let Err(error) = eval_res {
            self.recycle_batch_rows(rows);
            return Err(error);
        }
        self.record_batch_row_witness(&row_caps_before, &rows);
        if let Err(error) = check() {
            self.recycle_batch_rows(rows);
            return Err(error);
        }
        if super::verification::invalid_batched_logits(&self.decoder, &rows, block.len()) {
            self.recycle_batch_rows(rows);
            return Err(invalid("folded batched decoder returned invalid logits"));
        }

        if anchor_len == 1 {
            let prev = std::mem::replace(&mut self.logits, Arc::new(std::mem::take(&mut rows[0])));
            self.recycle_logits(prev);
        }

        let mut accepted = 0_usize;
        for (index, &candidate) in draft.iter().enumerate() {
            let target = self.sampled_suffix_token(check, effects.sampler)?;
            let stopped = options.stop_tokens.contains(&target);
            if stopped || target != candidate {
                let kept = anchor_len + accepted;
                self.decoder.commit_folded_verification(kept)?;
                carried_tokens.clear();
                carried_tokens.extend_from_slice(&block[..kept]);
                if kept > 0 {
                    stats.record_folded_carry(kept);
                }
                self.recycle_batch_rows(rows);
                if !stopped {
                    self.tokens.push(target);
                    *pending_anchor = Some(target);
                    Self::sampled_suffix_emit(target, check, &mut effects.emit)?;
                }
                let emitted = accepted + usize::from(!stopped);
                stats.record_verified_round(VerifiedRound {
                    proposed: draft.len(),
                    accepted,
                    verified: draft.len(),
                    target_evaluations: block.len(),
                    rolled_back: false,
                    bonus_tokens: 0,
                    correction_tokens: u64::from(!stopped),
                    fallback_tokens: 0,
                    replay_tokens: 0,
                });
                gate.record_verify_round(
                    draft.len(),
                    accepted,
                    emitted,
                    verify_start.elapsed(),
                    std::time::Duration::ZERO,
                );
                return Ok((SampledRound { emitted, stopped }, accepted));
            }
            self.tokens.push(candidate);
            let prev = std::mem::replace(
                &mut self.logits,
                Arc::new(std::mem::take(&mut rows[anchor_len + index])),
            );
            self.recycle_logits(prev);
            accepted += 1;
            Self::sampled_suffix_emit(candidate, check, &mut effects.emit)?;
        }

        let kept = block.len();
        self.decoder.commit_folded_verification(kept)?;
        carried_tokens.clear();
        carried_tokens.extend_from_slice(block);
        self.recycle_batch_rows(rows);
        stats.record_verified_round(VerifiedRound {
            proposed: draft.len(),
            accepted,
            verified: draft.len(),
            target_evaluations: block.len(),
            rolled_back: false,
            bonus_tokens: 0,
            correction_tokens: 0,
            fallback_tokens: 0,
            replay_tokens: 0,
        });
        gate.record_verify_round(
            draft.len(),
            accepted,
            draft.len(),
            verify_start.elapsed(),
            std::time::Duration::ZERO,
        );
        Ok((
            SampledRound {
                emitted: draft.len(),
                stopped: false,
            },
            accepted,
        ))
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
            return Err(invalid(
                "native sampler returned an out-of-vocabulary token",
            ));
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
        stats.record_verified_round(VerifiedRound {
            proposed: 0,
            accepted: 0,
            verified: 0,
            target_evaluations: 1,
            rolled_back: false,
            bonus_tokens: 0,
            correction_tokens: 0,
            fallback_tokens: 1,
            replay_tokens: 0,
        });
        Self::sampled_suffix_emit(token, check, &mut effects.emit)?;
        Ok(false)
    }

    fn prepare_sampled_suffix_batch(
        &mut self,
        draft: &[u32],
        check: &impl Fn() -> Result<()>,
        reserved: usize,
    ) -> Result<(usize, PreparedSuffixBatch<D::Checkpoint>)> {
        if draft.len() < 2 || draft.len() > self.decoder.batched_evaluation_capacity() {
            return Ok((0, None));
        }
        let cap_before = self.tokens.capacity();
        self.tokens.try_reserve(draft.len())?;
        if self.tokens.capacity() > cap_before {
            self.reallocation_witness.growth_reallocations = self
                .reallocation_witness
                .growth_reallocations
                .saturating_add(1);
        }
        let checkpoint = self.capture_transient(reserved, check)?;
        let mut rows = self.take_batch_rows(draft.len())?;
        let mut row_caps_before = [0_usize; MAX_SPECULATIVE_DRAFT_TOKENS];
        for (dst, row) in row_caps_before.iter_mut().zip(&rows) {
            *dst = row.capacity();
        }
        check()?;
        self.decoder.evaluate_batch_into(draft, &mut rows, check)?;
        self.record_batch_row_witness(&row_caps_before, &rows);
        check()?;
        if super::verification::invalid_batched_logits(&self.decoder, &rows, draft.len()) {
            self.recycle_snapshot(checkpoint);
            self.recycle_batch_rows(rows);
            return Err(invalid("batched decoder returned invalid logits"));
        }
        Ok((draft.len(), Some((checkpoint, rows))))
    }

    fn sampled_suffix_round_with_outcome(
        &mut self,
        draft: &[u32],
        options: &Generation,
        check: &impl Fn() -> Result<()>,
        effects: &mut SampledSuffixEffects<'_, impl Sampler, impl FnMut(u32) -> Result<()>>,
        stats: &mut SpeculativeStats,
        reserved: usize,
    ) -> Result<(SampledRound, usize)> {
        let (batch_evals, mut batch) = self.prepare_sampled_suffix_batch(draft, check, reserved)?;
        let mut sequential_evals = 0_usize;
        let mut accepted = 0_usize;
        let verified = usize::from(batch.is_some()) * draft.len();
        let mut seq_verified = 0_usize;
        for (index, &candidate) in draft.iter().enumerate() {
            if batch.is_none() {
                seq_verified += 1;
            }
            let target = self.sampled_suffix_token(check, effects.sampler)?;
            let stopped = options.stop_tokens.contains(&target);
            if stopped || target != candidate {
                let (rolled_back, mut replay_tokens) = if let Some((checkpoint, rows)) = batch {
                    self.recycle_batch_rows(rows);
                    self.sampled_suffix_restore(checkpoint, &draft[..index], check)?;
                    sequential_evals += index;
                    (true, index)
                } else {
                    (false, 0_usize)
                };
                if !stopped {
                    self.evaluate(target, check)?;
                    sequential_evals += 1;
                    if rolled_back {
                        replay_tokens += 1;
                    }
                    Self::sampled_suffix_emit(target, check, &mut effects.emit)?;
                }
                stats.record_verified_round(VerifiedRound {
                    proposed: draft.len(),
                    accepted,
                    verified: verified.max(seq_verified),
                    target_evaluations: batch_evals + sequential_evals,
                    rolled_back,
                    bonus_tokens: 0,
                    correction_tokens: u64::from(!stopped),
                    fallback_tokens: 0,
                    replay_tokens,
                });
                return Ok((
                    SampledRound {
                        emitted: index + usize::from(!stopped),
                        stopped,
                    },
                    accepted,
                ));
            }
            if let Some((_, rows)) = &mut batch {
                self.tokens.push(candidate);
                let previous_logits =
                    std::mem::replace(&mut self.logits, Arc::new(std::mem::take(&mut rows[index])));
                self.recycle_logits(previous_logits);
            } else {
                self.evaluate(candidate, check)?;
                sequential_evals += 1;
            }
            accepted += 1;
            Self::sampled_suffix_emit(candidate, check, &mut effects.emit)?;
        }
        if let Some((checkpoint, rows)) = batch {
            self.recycle_snapshot(checkpoint);
            self.recycle_batch_rows(rows);
        }
        stats.record_verified_round(VerifiedRound {
            proposed: draft.len(),
            accepted,
            verified: verified.max(seq_verified),
            target_evaluations: batch_evals + sequential_evals,
            rolled_back: false,
            bonus_tokens: 0,
            correction_tokens: 0,
            fallback_tokens: 0,
            replay_tokens: 0,
        });
        Ok((
            SampledRound {
                emitted: draft.len(),
                stopped: false,
            },
            accepted,
        ))
    }

    fn sampled_suffix_restore(
        &mut self,
        checkpoint: Snapshot<D::Checkpoint>,
        accepted: &[u32],
        check: &impl Fn() -> Result<()>,
    ) -> Result<()> {
        let Snapshot {
            state,
            tokens,
            media_spans,
            logits,
            ..
        } = checkpoint;
        self.faulted = true;
        let restored = self.decoder.restore(&state);
        self.tokens = tokens;
        self.media_spans = media_spans;
        let previous_logits = std::mem::replace(&mut self.logits, logits);
        restored?;
        self.faulted = false;
        self.recycle_checkpoint(state);
        self.recycle_logits(previous_logits);
        check()?;
        for &token in accepted {
            self.evaluate(token, check)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::Limits;
    use crate::speculative::SuffixDrafter;
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

        fn identity(&self) -> &'static str {
            "sampled-suffix-hybrid-control"
        }
        fn vocab_size(&self) -> usize {
            4
        }
        fn checkpoint_bytes(&self) -> usize {
            256
        }
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
            if self.restores == 1
                && let Some(cancellation) = &self.cancel_replay
            {
                cancellation.cancel();
            }
            self.state.tokens.push(token);
            self.state.recurrent = self
                .state
                .recurrent
                .wrapping_mul(17)
                .wrapping_add(u64::from(token) + 1);
            self.state.convolution.rotate_left(1);
            self.state.convolution[2] = token;
            let value = f32::from(u8::try_from(self.state.recurrent & 255)?);
            Ok(vec![value, value * 0.5, -value, value * -0.5])
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
            if self.malformed {
                let last = rows.len() - 1;
                rows[last][0] = f32::NAN;
            }
            Ok(())
        }
        fn checkpoint(&self) -> Result<Self::Checkpoint> {
            Ok(self.state.clone())
        }
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
            self.observed
                .push(logits.iter().map(|value| value.to_bits()).collect());
            let token = self.script[self.draws];
            self.draws += 1;
            if let Some(signal) = &self.signal {
                signal.set(self.draws);
            }
            if let Some(cancellation) = &self.cancel {
                cancellation.cancel();
            }
            Ok(token)
        }
    }

    fn runtime(capacity: usize) -> Result<Runtime<BatchControl>> {
        Runtime::new(
            BatchControl {
                capacity,
                ..BatchControl::default()
            },
            Limits {
                context_tokens: 64,
                checkpoints: 0,
                checkpoint_bytes: 65_536,
            },
        )
    }

    fn suffix() -> Result<SuffixDraftConfig> {
        let config = SuffixDraftConfig::new(1, 8, 4)?;
        // Repeated single tokens select a more recent one-token overlapping
        // continuation. This period-four history really proposes four tokens.
        assert_eq!(SuffixDrafter::propose(&PROMPT, config, 4)?, MATCHED[..4]);
        Ok(config)
    }

    #[test]
    fn sampled_suffix_matches_target_draws_logits_and_full_state_at_every_rejection() -> Result<()>
    {
        for capacity in [0, 8] {
            for rejected in 0..=4 {
                let mut ordinary = runtime(capacity)?;
                let mut speculative = runtime(capacity)?;
                let mut script = (0..16).map(|index| MATCHED[index % 4]).collect::<Vec<_>>();
                if rejected < 4 {
                    script[rejected] = 3;
                }
                let mut expected_sampler = ScriptSampler {
                    script,
                    ..ScriptSampler::default()
                };
                let mut actual_sampler = expected_sampler.clone();
                let mut expected = Vec::new();
                let mut actual = Vec::new();
                let options = Generation {
                    max_tokens: 16,
                    stop_tokens: Vec::new(),
                };
                let deadline = Instant::now() + Duration::from_secs(60);
                let expected_completion = ordinary.generate_sampled_until(
                    &PROMPT,
                    &options,
                    Prefill::default(),
                    &Cancellation::default(),
                    deadline,
                    &mut expected_sampler,
                    |token| {
                        expected.push(token);
                        Ok(())
                    },
                )?;
                let (actual_completion, stats) = speculative.generate_sampled_speculative_until(
                    &PROMPT,
                    &options,
                    Prefill::default(),
                    suffix()?,
                    &Cancellation::default(),
                    deadline,
                    &mut actual_sampler,
                    |token| {
                        actual.push(token);
                        Ok(())
                    },
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
            let mut expected_sampler = ScriptSampler {
                script,
                ..ScriptSampler::default()
            };
            let mut actual_sampler = expected_sampler.clone();
            let mut expected = Vec::new();
            let mut actual = Vec::new();
            let options = Generation {
                max_tokens: 6,
                stop_tokens: vec![3],
            };
            let deadline = Instant::now() + Duration::from_secs(60);
            let expected_completion = ordinary.generate_sampled_until(
                &PROMPT,
                &options,
                Prefill::default(),
                &Cancellation::default(),
                deadline,
                &mut expected_sampler,
                |token| {
                    expected.push(token);
                    Ok(())
                },
            )?;
            let (actual_completion, _) = speculative.generate_sampled_speculative_until(
                &PROMPT,
                &options,
                Prefill::default(),
                suffix()?,
                &Cancellation::default(),
                deadline,
                &mut actual_sampler,
                |token| {
                    actual.push(token);
                    Ok(())
                },
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
        let mut sampler = ScriptSampler {
            script: vec![0],
            ..ScriptSampler::default()
        };
        let (completion, stats) = speculative.generate_sampled_speculative_until(
            &PROMPT,
            &Generation {
                max_tokens: 4,
                stop_tokens: vec![0],
            },
            Prefill::default(),
            suffix()?,
            &Cancellation::default(),
            Instant::now() + Duration::from_secs(60),
            &mut sampler,
            |_| Err(invalid("a sampled stop must never be delivered")),
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
            let mut expected_sampler = ReferenceSampler::new(
                Sampling {
                    temperature: 80.0,
                    top_k: 4,
                    top_p: 1.0,
                },
                seed,
            )?;
            let mut actual_sampler = expected_sampler.clone();
            let mut expected = Vec::new();
            let mut actual = Vec::new();
            let options = Generation {
                max_tokens: 6,
                stop_tokens: Vec::new(),
            };
            let deadline = Instant::now() + Duration::from_secs(60);
            let expected_completion = ordinary.generate_sampled_until(
                &PROMPT,
                &options,
                Prefill::default(),
                &Cancellation::default(),
                deadline,
                &mut expected_sampler,
                |token| {
                    expected.push(token);
                    Ok(())
                },
            )?;
            let (actual_completion, _) = speculative.generate_sampled_speculative_until(
                &PROMPT,
                &options,
                Prefill::default(),
                suffix()?,
                &Cancellation::default(),
                deadline,
                &mut actual_sampler,
                |token| {
                    actual.push(token);
                    Ok(())
                },
            )?;
            assert_eq!(actual, expected);
            assert_eq!(actual_completion, expected_completion);
            assert_eq!(speculative.decoder.state, ordinary.decoder.state);
            assert_eq!(speculative.logits, ordinary.logits);
            for _ in 0..10 {
                assert_eq!(
                    actual_sampler.sample(&[0.0; 4])?,
                    expected_sampler.sample(&[0.0; 4])?
                );
            }
        }
        Ok(())
    }

    #[test]
    fn sampled_suffix_failures_restore_caller_rng_and_request_or_latch_failed_restore() -> Result<()>
    {
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
            let options = Generation {
                max_tokens: 4,
                stop_tokens: Vec::new(),
            };
            let mut delivered = 0;
            let result = speculative.generate_sampled_speculative_until(
                &PROMPT,
                &options,
                Prefill::default(),
                suffix()?,
                &cancellation,
                Instant::now() + Duration::from_secs(60),
                &mut sampler,
                |_| {
                    delivered += 1;
                    if failure == 0 {
                        return Err(invalid("injected callback failure"));
                    }
                    if failure == 1 {
                        cancellation.cancel();
                    }
                    Ok(())
                },
            );
            assert!(result.is_err());
            assert_eq!(sampler.draws, 0);
            assert_eq!(sampler.observed, Vec::<Vec<u32>>::new());
            assert_eq!(speculative.tokens, Vec::<u32>::new());
            assert!(speculative.logits.is_empty());
            assert_eq!(speculative.is_faulted(), failure == 4);
            if failure == 4 {
                assert_eq!(speculative.decoder.restores, 1);
            } else {
                assert_eq!(speculative.decoder.state, HybridState::default());
            }
            if failure == 3 {
                assert_eq!(delivered, 0);
            }
            if failure == 5 || failure == 6 {
                assert_eq!(speculative.decoder.restores, 2);
            }
            if failure != 4 {
                speculative.decoder.malformed = false;
                speculative.decoder.fail_replay = false;
                speculative.decoder.cancel_replay = None;
                sampler.cancel = None;
                let _ = speculative.generate_sampled_speculative_until(
                    &PROMPT,
                    &options,
                    Prefill::default(),
                    suffix()?,
                    &Cancellation::default(),
                    Instant::now() + Duration::from_secs(60),
                    &mut sampler,
                    |_| Ok(()),
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
        let mut sampler = ScriptSampler {
            script: vec![0; 4],
            ..ScriptSampler::default()
        };
        let mut delivered = 0;
        let result = speculative.generate_sampled_speculative_until(
            &PROMPT,
            &Generation {
                max_tokens: 4,
                stop_tokens: Vec::new(),
            },
            Prefill::default(),
            suffix()?,
            &Cancellation::default(),
            Instant::now() + Duration::from_secs(60),
            &mut sampler,
            |_| {
                delivered += 1;
                Ok(())
            },
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
            script: vec![0; 4],
            signal: Some(draws.clone()),
            ..ScriptSampler::default()
        };
        let mut delivered = 0;
        let result = speculative.generate_sampled_speculative_checked(
            &PROMPT,
            &Generation {
                max_tokens: 4,
                stop_tokens: Vec::new(),
            },
            Prefill::default(),
            suffix()?,
            AdaptiveGateConfig::default(),
            &|| {
                if draws.get() == 0 {
                    Ok(())
                } else {
                    Err(std::io::Error::new(
                        std::io::ErrorKind::TimedOut,
                        "injected post-draw deadline",
                    )
                    .into())
                }
            },
            &mut sampler,
            |_| {
                delivered += 1;
                Ok(())
            },
        );
        assert!(result.is_err());
        assert_eq!(sampler.draws, 0);
        assert_eq!(delivered, 0);
        assert_eq!(speculative.decoder.state, HybridState::default());
        assert!(!speculative.is_faulted());
        Ok(())
    }

    fn advance_hybrid(state: &mut HybridState, token: u32) -> Result<Vec<f32>> {
        state.tokens.push(token);
        state.recurrent = state
            .recurrent
            .wrapping_mul(17)
            .wrapping_add(u64::from(token) + 1);
        state.convolution.rotate_left(1);
        state.convolution[2] = token;
        let value = f32::from(u8::try_from(state.recurrent & 255)?);
        Ok(vec![value, value * 0.5, -value, value * -0.5])
    }

    #[derive(Clone, Debug)]
    struct FoldedBatchControl {
        kv_state: HybridState,
        recurrent_state: HybridState,
        pending: Vec<u32>,
        carried: Vec<u32>,
        capacity: usize,
        folded_blocks: usize,
        flushes: usize,
        serial_evals: usize,
        block_delay: Duration,
        serial_delay: Duration,
    }

    impl FoldedBatchControl {
        fn new(capacity: usize) -> Self {
            Self {
                kv_state: HybridState::default(),
                recurrent_state: HybridState::default(),
                pending: Vec::new(),
                carried: Vec::new(),
                capacity,
                folded_blocks: 0,
                flushes: 0,
                serial_evals: 0,
                block_delay: Duration::ZERO,
                serial_delay: Duration::ZERO,
            }
        }
    }

    #[derive(Clone, Debug, PartialEq)]
    struct FoldedCheckpoint {
        kv_state: HybridState,
        recurrent_state: HybridState,
    }

    impl Decoder for FoldedBatchControl {
        type Checkpoint = FoldedCheckpoint;

        fn identity(&self) -> &'static str {
            "sampled-suffix-folded-control"
        }

        fn vocab_size(&self) -> usize {
            4
        }

        fn reset(&mut self) -> Result<()> {
            self.discard_folded_verification();
            self.kv_state = HybridState::default();
            self.recurrent_state = HybridState::default();
            Ok(())
        }

        fn evaluate(&mut self, token: u32) -> Result<Vec<f32>> {
            if self.has_pending_folded_verification() {
                return Err(invalid("pending folded verification forbids evaluate"));
            }
            if !self.serial_delay.is_zero() {
                std::thread::sleep(self.serial_delay);
            }
            self.serial_evals += 1;
            let _ = advance_hybrid(&mut self.recurrent_state, token)?;
            advance_hybrid(&mut self.kv_state, token)
        }

        fn evaluate_into(&mut self, token: u32, output: &mut Vec<f32>) -> Result<()> {
            let logits = self.evaluate(token)?;
            output.clear();
            output.extend_from_slice(&logits);
            Ok(())
        }

        fn batched_evaluation_capacity(&self) -> usize {
            self.capacity
        }

        fn supports_folded_verification(&self) -> bool {
            true
        }

        fn has_pending_folded_verification(&self) -> bool {
            !self.pending.is_empty() || !self.carried.is_empty()
        }

        fn evaluate_verification_block_folded(
            &mut self,
            carried_tokens: &[u32],
            tokens: &[u32],
            output: &mut Vec<Vec<f32>>,
            _check: &dyn Fn() -> Result<()>,
        ) -> Result<()> {
            if !self.pending.is_empty() {
                return Err(invalid("folded block already pending commit"));
            }
            if self.carried.as_slice() != carried_tokens {
                return Err(invalid("folded carry mismatch"));
            }
            if !self.block_delay.is_zero() {
                std::thread::sleep(self.block_delay);
            }
            self.folded_blocks += 1;
            for &tok in &self.carried {
                let _ = advance_hybrid(&mut self.recurrent_state, tok)?;
            }
            self.carried.clear();
            output.clear();
            for &tok in tokens {
                output.push(advance_hybrid(&mut self.kv_state, tok)?);
            }
            self.pending.extend_from_slice(tokens);
            Ok(())
        }

        fn commit_folded_verification(&mut self, kept: usize) -> Result<()> {
            let pending_len = self.pending.len();
            if pending_len == 0 || kept > pending_len {
                return Err(invalid("invalid folded commit count"));
            }
            let trim = pending_len - kept;
            if trim > 0 {
                let keep_total = self.kv_state.tokens.len() - trim;
                let kept_tokens = self.kv_state.tokens[..keep_total].to_vec();
                self.kv_state = HybridState::default();
                for tok in kept_tokens {
                    let _ = advance_hybrid(&mut self.kv_state, tok)?;
                }
            }
            self.carried.clear();
            self.carried.extend_from_slice(&self.pending[..kept]);
            self.pending.clear();
            Ok(())
        }

        fn flush_folded_verification(&mut self, _check: &dyn Fn() -> Result<()>) -> Result<()> {
            if !self.pending.is_empty() {
                return Err(invalid("cannot flush uncommitted folded block"));
            }
            if self.carried.is_empty() {
                return Ok(());
            }
            self.flushes += 1;
            for &tok in &self.carried {
                let _ = advance_hybrid(&mut self.recurrent_state, tok)?;
            }
            self.carried.clear();
            Ok(())
        }

        fn discard_folded_verification(&mut self) {
            self.pending.clear();
            self.carried.clear();
        }

        fn checkpoint_bytes(&self) -> usize {
            64
        }

        fn checkpoint(&self) -> Result<Self::Checkpoint> {
            if self.has_pending_folded_verification() {
                return Err(invalid("cannot checkpoint with pending folded state"));
            }
            Ok(FoldedCheckpoint {
                kv_state: self.kv_state.clone(),
                recurrent_state: self.recurrent_state.clone(),
            })
        }

        fn checkpoint_into(&self, target: &mut Self::Checkpoint) -> Result<()> {
            *target = self.checkpoint()?;
            Ok(())
        }

        fn restore(&mut self, checkpoint: &Self::Checkpoint) -> Result<()> {
            self.discard_folded_verification();
            self.kv_state = checkpoint.kv_state.clone();
            self.recurrent_state = checkpoint.recurrent_state.clone();
            Ok(())
        }
    }

    #[test]
    fn folded_speculation_preserves_exact_draws_logits_state_and_bridges_cursor() -> Result<()> {
        for rejected in 0..=4 {
            let limits = Limits {
                context_tokens: 128,
                checkpoints: 4,
                checkpoint_bytes: 1024,
            };
            let mut ordinary = Runtime::new(FoldedBatchControl::new(0), limits)?;
            let mut folded = Runtime::new(FoldedBatchControl::new(8), limits)?;
            let mut script = (0..16).map(|index| MATCHED[index % 4]).collect::<Vec<_>>();
            if rejected < 4 {
                script[rejected] = 3;
            }
            let mut expected_sampler = ScriptSampler {
                script,
                ..ScriptSampler::default()
            };
            let mut actual_sampler = expected_sampler.clone();
            let mut expected = Vec::new();
            let mut actual = Vec::new();
            let options = Generation {
                max_tokens: 16,
                stop_tokens: Vec::new(),
            };
            let gate_cfg = AdaptiveGateConfig {
                enabled: false,
                ..AdaptiveGateConfig::default()
            };
            let deadline = Instant::now() + Duration::from_secs(60);
            let exp_comp = ordinary.generate_sampled_until(
                &PROMPT,
                &options,
                Prefill::default(),
                &Cancellation::default(),
                deadline,
                &mut expected_sampler,
                |t| {
                    expected.push(t);
                    Ok(())
                },
            )?;
            let (act_comp, stats) = folded.generate_sampled_speculative_with_gate_until(
                &PROMPT,
                &options,
                Prefill::default(),
                suffix()?,
                gate_cfg,
                &Cancellation::default(),
                deadline,
                &mut actual_sampler,
                |t| {
                    actual.push(t);
                    Ok(())
                },
            )?;
            assert_eq!(act_comp, exp_comp);
            assert_eq!(actual, expected);
            assert_eq!(actual_sampler.draws, expected_sampler.draws);
            assert_eq!(actual_sampler.observed, expected_sampler.observed);
            assert_eq!(folded.tokens, ordinary.tokens);
            assert_eq!(folded.logits, ordinary.logits);
            assert_eq!(folded.decoder.kv_state, ordinary.decoder.kv_state);
            assert_eq!(
                folded.decoder.recurrent_state,
                ordinary.decoder.recurrent_state
            );
            assert!(!folded.decoder.has_pending_folded_verification());
            assert_eq!(stats.replay_steps, 0);
            assert_eq!(stats.rollback_restores, 0);
            assert!(folded.decoder.folded_blocks > 0);
            if rejected < 4 {
                assert!(stats.cursor_draft_tokens > 0);
                if rejected > 0 {
                    assert!(stats.folded_replay_steps > 0);
                }
            }
        }
        Ok(())
    }

    #[test]
    fn folded_adaptive_gate_pauses_low_acceptance_prose_into_serial_burst() -> Result<()> {
        let limits = Limits {
            context_tokens: 128,
            checkpoints: 4,
            checkpoint_bytes: 1024,
        };
        let mut seq = FoldedBatchControl::new(0);
        seq.serial_delay = Duration::from_micros(200);
        let mut ordinary = Runtime::new(seq, limits)?;

        let mut spec = FoldedBatchControl::new(8);
        spec.serial_delay = Duration::from_micros(200);
        spec.block_delay = Duration::from_micros(900);
        let mut gated = Runtime::new(spec, limits)?;

        let script = vec![3, 1, 2, 3, 2, 1, 3, 2, 1, 0, 3, 2, 1, 0, 2, 3, 1, 2, 0, 1];
        let mut exp_sampler = ScriptSampler {
            script,
            ..ScriptSampler::default()
        };
        let mut act_sampler = exp_sampler.clone();
        let options = Generation {
            max_tokens: 20,
            stop_tokens: Vec::new(),
        };
        let gate_cfg = AdaptiveGateConfig {
            enabled: true,
            warmup_rounds: 2,
            smoothing: 0.5,
            initial_verify_to_serial_ratio: 2.5,
            break_even_margin: 1.0,
            min_burst_tokens: 4,
            max_burst_tokens: 8,
        };
        let deadline = Instant::now() + Duration::from_secs(60);
        let mut expected = Vec::new();
        let mut actual = Vec::new();
        let _ = ordinary.generate_sampled_until(
            &PROMPT,
            &options,
            Prefill::default(),
            &Cancellation::default(),
            deadline,
            &mut exp_sampler,
            |t| {
                expected.push(t);
                Ok(())
            },
        )?;
        let (_, stats) = gated.generate_sampled_speculative_with_gate_until(
            &PROMPT,
            &options,
            Prefill::default(),
            suffix()?,
            gate_cfg,
            &Cancellation::default(),
            deadline,
            &mut act_sampler,
            |t| {
                actual.push(t);
                Ok(())
            },
        )?;
        assert_eq!(actual, expected);
        assert_eq!(act_sampler.observed, exp_sampler.observed);
        assert_eq!(gated.decoder.kv_state, ordinary.decoder.kv_state);
        assert_eq!(
            gated.decoder.recurrent_state,
            ordinary.decoder.recurrent_state
        );
        assert!(stats.adaptive_gate_skips > 0);
        assert!(stats.serial_burst_tokens > 0);
        Ok(())
    }
}
