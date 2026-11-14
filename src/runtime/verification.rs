//! Opt-in weight-sharing target verification with complete-prefix replay.

use super::{
    Arc, Decoder, FiniteLogits, Result, Runtime, Snapshot, SpeculativeStats, VerificationLimits,
    VerifiedRound, greedy_finite, invalid,
};

impl<D: Decoder> Runtime<D> {
    pub(super) fn verify_batched_draft(
        &mut self,
        draft: &[u32],
        limits: VerificationLimits<'_>,
        rollback: &Snapshot<D::Checkpoint>,
        check: &impl Fn() -> Result<()>,
    ) -> Result<(Vec<u32>, SpeculativeStats)> {
        let allow_bonus = draft.len() < limits.max_new_tokens;
        let max_advance = draft.len() + usize::from(allow_bonus);
        let mut committed = Vec::new();
        committed.try_reserve_exact(max_advance)?;
        let cap_before = self.tokens.capacity();
        self.tokens.try_reserve(max_advance)?;
        if self.tokens.capacity() > cap_before {
            self.reallocation_witness.growth_reallocations = self
                .reallocation_witness
                .growth_reallocations
                .saturating_add(1);
        }
        let mut rows = self.take_batch_rows(draft.len())?;
        let mut row_caps_before = [0_usize; crate::speculative::MAX_SPECULATIVE_DRAFT_TOKENS];
        for (dst, row) in row_caps_before.iter_mut().zip(&rows) {
            *dst = row.capacity();
        }
        check()?;
        self.decoder.evaluate_batch_into(draft, &mut rows, check)?;
        self.record_batch_row_witness(&row_caps_before, &rows);
        check()?;
        // Decoder is public: independently validate every output before using
        // any row to accept tokens, including rows beyond a rejected prefix.
        if rows.len() != draft.len()
            || rows.iter().any(|row| {
                row.len() != self.decoder.vocab_size()
                    || row.is_empty()
                    || row.len() > 1_048_576
                    || !crate::kernels::is_finite_slice(row)
            })
        {
            self.recycle_batch_rows(rows);
            return Err(invalid("batched decoder returned invalid logits"));
        }
        let mut stats = SpeculativeStats::default();
        for (index, &candidate) in draft.iter().enumerate() {
            check()?;
            let logits = if index == 0 {
                &self.logits[..]
            } else {
                &rows[index - 1]
            };
            let expected = greedy_finite(FiniteLogits::from_validated(logits)?)?;
            let stopped = limits.stop_tokens.contains(&expected);
            if stopped || candidate != expected {
                // The batched graph contains the complete proposed suffix.
                // Restore once, then replay only the accepted prefix before
                // the correction. KV-only truncation cannot recover GDN/conv.
                self.recycle_batch_rows(rows);
                self.faulted = true;
                self.decoder.restore(&rollback.state)?;
                self.faulted = false;
                check()?;
                for &accepted in &draft[..index] {
                    self.evaluate(accepted, check)?;
                    committed.push(accepted);
                }
                if !stopped {
                    self.evaluate(expected, check)?;
                    committed.push(expected);
                }
                let replay_tokens = index + usize::from(!stopped);
                stats.record_verified_round(VerifiedRound {
                    proposed: draft.len(),
                    accepted: index,
                    verified: draft.len(),
                    target_evaluations: draft.len() + replay_tokens,
                    rolled_back: true,
                    bonus_tokens: 0,
                    correction_tokens: u64::from(!stopped),
                    fallback_tokens: 0,
                    replay_tokens,
                });
                return Ok((committed, stats));
            }
        }
        // All proposed state is already present. Publish its final distribution
        // and advance at most one bonus token when limits allow.
        self.tokens.extend_from_slice(draft);
        let final_row = rows
            .pop()
            .ok_or_else(|| invalid("missing final batch logits"))?;
        self.recycle_batch_rows(rows);
        let previous_logits = std::mem::replace(&mut self.logits, Arc::new(final_row));
        self.recycle_logits(previous_logits);
        committed.extend_from_slice(draft);
        let mut bonus_emitted = 0_u64;
        if allow_bonus {
            let bonus = greedy_finite(FiniteLogits::from_validated(&self.logits)?)?;
            if !limits.stop_tokens.contains(&bonus) {
                self.evaluate(bonus, check)?;
                committed.push(bonus);
                bonus_emitted = 1;
            }
        }
        stats.record_verified_round(VerifiedRound {
            proposed: draft.len(),
            accepted: draft.len(),
            verified: draft.len(),
            target_evaluations: draft.len() + usize::from(bonus_emitted > 0),
            rolled_back: false,
            bonus_tokens: bonus_emitted,
            correction_tokens: 0,
            fallback_tokens: 0,
            replay_tokens: 0,
        });
        Ok((committed, stats))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::{Cancellation, Generation, Limits};
    use std::cell::Cell;

    #[derive(Default)]
    struct BatchControl {
        tokens: Vec<u32>,
        admitted: Cell<usize>,
        restores: usize,
        malformed: bool,
        fail_restore: bool,
        fail_replay: bool,
        replay_steps: usize,
    }

    impl Decoder for BatchControl {
        type Checkpoint = Vec<u32>;

        fn identity(&self) -> &'static str {
            "batched-verification-control"
        }
        fn vocab_size(&self) -> usize {
            2
        }
        fn checkpoint_bytes(&self) -> usize {
            256
        }
        fn admit_work(&self, _: usize, evaluations: usize) -> Result<()> {
            self.admitted.set(evaluations);
            Ok(())
        }
        fn reset(&mut self) -> Result<()> {
            self.tokens.clear();
            Ok(())
        }
        fn evaluate(&mut self, token: u32) -> Result<Vec<f32>> {
            if self.fail_replay && self.restores > 0 {
                self.replay_steps += 1;
                if self.replay_steps == 2 {
                    return Err(invalid("replay failed"));
                }
            }
            self.tokens.push(token);
            Ok(if token == 0 {
                vec![2.0, 0.0]
            } else {
                vec![0.0, 2.0]
            })
        }
        fn batched_evaluation_capacity(&self) -> usize {
            8
        }
        fn evaluate_batch_into(
            &mut self,
            tokens: &[u32],
            rows: &mut Vec<Vec<f32>>,
            check: &dyn Fn() -> Result<()>,
        ) -> Result<()> {
            rows.clear();
            for &token in tokens {
                check()?;
                rows.push(self.evaluate(token)?);
            }
            if self.malformed {
                rows[1][0] = f32::NAN;
            }
            Ok(())
        }
        fn checkpoint(&self) -> Result<Self::Checkpoint> {
            Ok(self.tokens.clone())
        }
        fn restore(&mut self, state: &Self::Checkpoint) -> Result<()> {
            self.restores += 1;
            if self.fail_restore {
                return Err(invalid("restore failed"));
            }
            self.tokens.clone_from(state);
            Ok(())
        }
    }

    fn prepared() -> Result<Runtime<BatchControl>> {
        let mut runtime = Runtime::new(
            BatchControl::default(),
            Limits {
                context_tokens: 32,
                checkpoints: 0,
                checkpoint_bytes: 4096,
            },
        )?;
        runtime.generate(
            &[0],
            &Generation {
                max_tokens: 1,
                stop_tokens: Vec::new(),
            },
            &Cancellation::default(),
            |_| Ok(()),
        )?;
        Ok(runtime)
    }

    #[test]
    fn batched_verification_reserves_rejected_work_and_replay() -> Result<()> {
        let mut runtime = prepared()?;
        let (tokens, stats) = runtime.verify_draft_block(&[0, 0, 1], &Cancellation::default())?;
        assert_eq!(runtime.decoder.admitted.get(), 7);
        assert_eq!(tokens, [0, 0, 0]);
        assert_eq!(stats.draft_tokens_accepted, 2);
        assert_eq!(stats.target_evaluations, 6);
        assert_eq!(stats.rollback_restores, 1);
        assert_eq!(runtime.decoder.tokens, runtime.tokens);
        Ok(())
    }

    #[test]
    fn batched_verification_rejects_malformed_rows_and_restores_failed_replay() -> Result<()> {
        for malformed in [false, true] {
            let mut runtime = prepared()?;
            let before = runtime.tokens.clone();
            let logits = Arc::clone(&runtime.logits);
            runtime.decoder.malformed = malformed;
            runtime.decoder.fail_replay = !malformed;
            assert!(
                runtime
                    .verify_draft_block(&[0, 0, 1], &Cancellation::default())
                    .is_err()
            );
            assert_eq!(runtime.tokens, before);
            assert_eq!(runtime.decoder.tokens, before);
            assert_eq!(runtime.logits, logits);
            assert!(!runtime.is_faulted());
        }
        Ok(())
    }

    #[test]
    fn batched_verification_failed_prefix_restore_latches_without_retry() -> Result<()> {
        let mut runtime = prepared()?;
        runtime.decoder.fail_restore = true;
        assert!(
            runtime
                .verify_draft_block(&[0, 1], &Cancellation::default())
                .is_err()
        );
        assert_eq!(runtime.decoder.restores, 1);
        assert!(runtime.is_faulted());
        assert!(
            runtime
                .verify_draft_block(&[0, 1], &Cancellation::default())
                .is_err()
        );
        assert_eq!(runtime.decoder.restores, 1);
        Ok(())
    }
}
