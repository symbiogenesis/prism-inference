//! Opt-in weight-sharing target verification with complete-prefix replay.

use super::*;

impl<D: Decoder> Runtime<D> {
    pub(super) fn verify_batched_draft(
        &mut self,
        draft: &[u32],
        rollback: &Snapshot<D::Checkpoint>,
        check: &impl Fn() -> Result<()>,
    ) -> Result<(Vec<u32>, SpeculativeStats)> {
        let mut committed = Vec::new();
        committed.try_reserve_exact(draft.len() + 1)?;
        self.tokens.try_reserve(draft.len() + 1)?;
        let mut rows = Vec::new();
        check()?;
        self.decoder.evaluate_batch_into(draft, &mut rows, check)?;
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
            return Err(invalid("batched decoder returned invalid logits"));
        }
        let mut stats = SpeculativeStats {
            draft_rounds: 1,
            draft_tokens_proposed: draft.len(),
            draft_tokens_accepted: 0,
            target_evaluations: draft.len(),
            rollback_restores: 0,
        };
        for (index, &candidate) in draft.iter().enumerate() {
            check()?;
            let logits = if index == 0 { &self.logits[..] } else { &rows[index - 1] };
            let expected = greedy_finite(FiniteLogits::from_validated(logits)?)?;
            if candidate != expected {
                // The batched graph contains the complete proposed suffix.
                // Restore once, then replay only the accepted prefix before
                // the correction. KV-only truncation cannot recover GDN/conv.
                self.faulted = true;
                self.decoder.restore(&rollback.state)?;
                self.faulted = false;
                stats.rollback_restores = 1;
                check()?;
                for &accepted in &draft[..index] {
                    self.evaluate(accepted, check)?;
                    stats.target_evaluations += 1;
                    committed.push(accepted);
                }
                self.evaluate(expected, check)?;
                stats.target_evaluations += 1;
                committed.push(expected);
                return Ok((committed, stats));
            }
            stats.draft_tokens_accepted += 1;
        }
        // All proposed state is already present. Publish its final distribution
        // and advance precisely one bonus token from that distribution.
        self.tokens.extend_from_slice(draft);
        self.logits = Arc::new(rows.pop().ok_or_else(|| invalid("missing final batch logits"))?);
        committed.extend_from_slice(draft);
        let bonus = greedy_finite(FiniteLogits::from_validated(&self.logits)?)?;
        self.evaluate(bonus, check)?;
        stats.target_evaluations += 1;
        committed.push(bonus);
        Ok((committed, stats))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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

        fn identity(&self) -> &'static str { "batched-verification-control" }
        fn vocab_size(&self) -> usize { 2 }
        fn checkpoint_bytes(&self) -> usize { 256 }
        fn admit_work(&self, _: usize, evaluations: usize) -> Result<()> {
            self.admitted.set(evaluations);
            Ok(())
        }
        fn reset(&mut self) -> Result<()> { self.tokens.clear(); Ok(()) }
        fn evaluate(&mut self, token: u32) -> Result<Vec<f32>> {
            if self.fail_replay && self.restores > 0 {
                self.replay_steps += 1;
                if self.replay_steps == 2 { return Err(invalid("replay failed")); }
            }
            self.tokens.push(token);
            Ok(if token == 0 { vec![2.0, 0.0] } else { vec![0.0, 2.0] })
        }
        fn batched_evaluation_capacity(&self) -> usize { 8 }
        fn evaluate_batch_into(&mut self, tokens: &[u32], rows: &mut Vec<Vec<f32>>, check: &dyn Fn() -> Result<()>) -> Result<()> {
            rows.clear();
            for &token in tokens {
                check()?;
                rows.push(self.evaluate(token)?);
            }
            if self.malformed { rows[1][0] = f32::NAN; }
            Ok(())
        }
        fn checkpoint(&self) -> Result<Self::Checkpoint> { Ok(self.tokens.clone()) }
        fn restore(&mut self, state: &Self::Checkpoint) -> Result<()> {
            self.restores += 1;
            if self.fail_restore { return Err(invalid("restore failed")); }
            self.tokens.clone_from(state);
            Ok(())
        }
    }

    fn prepared() -> Result<Runtime<BatchControl>> {
        let mut runtime = Runtime::new(BatchControl::default(), Limits {
            context_tokens: 32, checkpoints: 0, checkpoint_bytes: 4096,
        })?;
        runtime.generate(&[0], &Generation { max_tokens: 1, stop_tokens: Vec::new() }, &Cancellation::default(), |_| Ok(()))?;
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
            assert!(runtime.verify_draft_block(&[0, 0, 1], &Cancellation::default()).is_err());
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
        assert!(runtime.verify_draft_block(&[0, 1], &Cancellation::default()).is_err());
        assert_eq!(runtime.decoder.restores, 1);
        assert!(runtime.is_faulted());
        assert!(runtime.verify_draft_block(&[0, 1], &Cancellation::default()).is_err());
        assert_eq!(runtime.decoder.restores, 1);
        Ok(())
    }
}
