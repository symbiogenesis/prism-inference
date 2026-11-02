//! Attribution must preserve successful and failed transaction semantics.

use super::*;
use bonsai_prism_runtime::runtime::{GenerationPhase, GenerationTimings};

#[test]
fn timing_preserves_sampled_output_state_rng_and_reused_prefix() -> Result<()> {
    let mut plain = Runtime::new(decoder(), LIMITS)?;
    let mut timed = Runtime::new(decoder(), LIMITS)?;
    let (mut plain_sampler, mut timed_sampler) =
        (SequenceSampler::default(), SequenceSampler::default());
    let mut timings = GenerationTimings::default();
    for attempt in 0..3 {
        let prompt = match attempt {
            0 => vec![1, 2, 3, 4],
            1 => plain.tokens().to_vec(),
            _ => vec![7, 6, 5],
        };
        let (mut plain_output, mut timed_output) = (Vec::new(), Vec::new());
        let deadline = Instant::now() + Duration::from_secs(60);
        let expected = plain.generate_sampled_until(
            &prompt,
            &options(),
            Prefill { block_tokens: 2 },
            &Cancellation::default(),
            deadline,
            &mut plain_sampler,
            |token| {
                plain_output.push(token);
                Ok(())
            },
        )?;
        let actual = timed.generate_sampled_until_with_timings(
            &prompt,
            &options(),
            Prefill { block_tokens: 2 },
            &Cancellation::default(),
            deadline,
            &mut timed_sampler,
            &mut timings,
            |token| {
                timed_output.push(token);
                Ok(())
            },
        )?;
        assert_eq!(actual, expected);
        assert_eq!(plain_output, timed_output);
        assert_eq!(plain.tokens(), timed.tokens());
        assert_eq!(plain.decoder().tokens, timed.decoder().tokens);
        assert_eq!(plain_sampler, timed_sampler);
        assert!(timings.entered && timings.succeeded);
        assert_eq!(timings.prompt_tokens, prompt.len());
        assert_eq!(timings.reused_tokens, Some(actual.reused_tokens));
        assert_eq!(
            timings.committed_generated_tokens,
            Some(actual.generated_tokens)
        );
        assert_eq!(
            timings
                .phase(GenerationPhase::PrefillIngestion)
                .completed_tokens_lower_bound,
            (prompt.len() - actual.reused_tokens).saturating_sub(1)
        );
        assert_eq!(
            timings
                .phase(GenerationPhase::PrefillEvaluation)
                .completed_tokens_lower_bound,
            usize::from(prompt.len() > actual.reused_tokens)
        );
        assert_eq!(
            timings
                .phase(GenerationPhase::DecodeEvaluation)
                .completed_tokens_lower_bound,
            1
        );
        assert_eq!(timings.phase(GenerationPhase::Callback).completed_calls, 1);
        assert_eq!(
            timings
                .phase(GenerationPhase::RollbackRestore)
                .attempted_calls,
            0
        );
        let phases: Duration = GenerationPhase::ALL
            .into_iter()
            .map(|phase| timings.phase(phase).elapsed)
            .sum();
        assert!(phases <= timings.total);
    }
    Ok(())
}

#[test]
fn timed_stop_draw_is_counted_without_decode_or_callback_work() -> Result<()> {
    let mut sampler = SequenceSampler::default();
    let mut expected_sampler = sampler.clone();
    let stop = expected_sampler.sample(&[0.0; 8])?;
    let mut runtime = Runtime::new(decoder(), LIMITS)?;
    let mut timings = GenerationTimings::default();
    let completion = runtime.generate_sampled_until_with_timings(
        &[1],
        &Generation {
            max_tokens: 2,
            stop_tokens: vec![stop],
        },
        Prefill::default(),
        &Cancellation::default(),
        Instant::now() + Duration::from_secs(60),
        &mut sampler,
        &mut timings,
        |_| Err(std::io::Error::other("stop must not emit").into()),
    )?;
    assert_eq!(completion.finish, Finish::Stop);
    assert_eq!(completion.generated_tokens, 0);
    assert_eq!(sampler, expected_sampler);
    assert_eq!(runtime.tokens(), [1]);
    assert_eq!(timings.phase(GenerationPhase::Sampling).completed_calls, 1);
    assert_eq!(
        timings
            .phase(GenerationPhase::DecodeEvaluation)
            .attempted_calls,
        0
    );
    assert_eq!(timings.phase(GenerationPhase::Callback).attempted_calls, 0);
    assert_eq!(timings.committed_generated_tokens, Some(0));
    Ok(())
}

#[test]
fn failed_callback_retains_provisional_work_and_rolls_back_state_and_rng() -> Result<()> {
    for restore_fails in [false, true] {
        let decoder = decoder();
        let failure = decoder.restore_fail.clone();
        let mut runtime = Runtime::new(decoder, LIMITS)?;
        let mut sampler = SequenceSampler::default();
        let before = sampler.clone();
        let mut timings = GenerationTimings::default();
        failure.set(restore_fails);
        assert!(
            runtime
                .generate_sampled_until_with_timings(
                    &[1, 2, 3],
                    &Generation {
                        max_tokens: 2,
                        stop_tokens: vec![]
                    },
                    Prefill { block_tokens: 2 },
                    &Cancellation::default(),
                    Instant::now() + Duration::from_secs(60),
                    &mut sampler,
                    &mut timings,
                    |_| Err(std::io::Error::other("failed delivery after possible output").into()),
                )
                .is_err()
        );
        assert_eq!(sampler, before);
        assert_eq!(runtime.tokens(), [0_u32; 0]);
        assert_eq!(runtime.is_faulted(), restore_fails);
        assert!(!timings.succeeded);
        assert_eq!(timings.committed_generated_tokens, None);
        assert_eq!(
            timings
                .phase(GenerationPhase::DecodeEvaluation)
                .completed_tokens_lower_bound,
            1
        );
        let callback = timings.phase(GenerationPhase::Callback);
        assert_eq!(callback.attempted_calls, 1);
        assert_eq!(callback.completed_calls, 0);
        assert!(callback.partial_progress_unknown);
        assert_eq!(
            timings
                .phase(GenerationPhase::RollbackRestore)
                .attempted_calls,
            1
        );
        assert_eq!(
            timings
                .phase(GenerationPhase::RollbackRestore)
                .completed_calls,
            usize::from(!restore_fails)
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
fn timed_cancel_after_successful_callback_never_reports_committed_output() -> Result<()> {
    let mut runtime = Runtime::new(decoder(), LIMITS)?;
    let mut sampler = SequenceSampler::default();
    let before = sampler.clone();
    let mut timings = GenerationTimings::default();
    let cancel = Cancellation::default();
    assert!(
        runtime
            .generate_sampled_until_with_timings(
                &[1],
                &options(),
                Prefill::default(),
                &cancel,
                Instant::now() + Duration::from_secs(60),
                &mut sampler,
                &mut timings,
                |_| {
                    cancel.cancel();
                    Ok(())
                },
            )
            .is_err()
    );
    assert_eq!(timings.phase(GenerationPhase::Callback).completed_calls, 1);
    assert_eq!(timings.committed_generated_tokens, None);
    assert!(!timings.succeeded);
    assert_eq!(sampler, before);
    assert_eq!(runtime.tokens(), [0_u32; 0]);
    // Reusing the receipt for a rejected attempt must not retain old work.
    assert!(
        runtime
            .generate_sampled_until_with_timings(
                &[1],
                &options(),
                Prefill::default(),
                &cancel,
                Instant::now(),
                &mut sampler,
                &mut timings,
                |_| Ok(()),
            )
            .is_err()
    );
    assert_eq!(timings.phase(GenerationPhase::Callback).attempted_calls, 0);
    assert_eq!(
        timings
            .phase(GenerationPhase::RollbackSnapshot)
            .attempted_calls,
        0
    );
    Ok(())
}

#[test]
fn failed_prefill_block_keeps_unknown_partial_work_distinct_from_zero() -> Result<()> {
    let decoder = state_only();
    decoder.fault.set(IngestFault::Failure);
    decoder.fault_token.set(Some(3));
    let operations = decoder.operations.clone();
    let mut runtime = Runtime::new(decoder, LIMITS)?;
    let mut timings = GenerationTimings::default();
    assert!(
        runtime
            .generate_sampled_until_with_timings(
                &[1, 2, 3, 4, 5],
                &options(),
                Prefill { block_tokens: 2 },
                &Cancellation::default(),
                Instant::now() + Duration::from_secs(60),
                &mut SequenceSampler::default(),
                &mut timings,
                |_| Ok(()),
            )
            .is_err()
    );
    assert_eq!(
        *operations.borrow(),
        [
            Operation::Ingest(1),
            Operation::Ingest(2),
            Operation::Ingest(3)
        ]
    );
    let prefill = timings.phase(GenerationPhase::PrefillIngestion);
    assert_eq!(prefill.attempted_tokens, 4);
    assert_eq!(prefill.completed_tokens_lower_bound, 2);
    assert!(prefill.partial_progress_unknown);
    assert_eq!(timings.phase(GenerationPhase::Sampling).attempted_calls, 0);
    assert_eq!(runtime.tokens(), [0_u32; 0]);
    Ok(())
}
