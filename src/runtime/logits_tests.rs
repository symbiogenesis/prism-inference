//! Immutable snapshot output and transactional buffer ownership controls.

use super::*;

#[derive(Clone, Copy)]
enum Failure {
    None,
    Nonfinite,
    WrongSize,
    Evaluate,
}

struct Reusable {
    tokens: Vec<u32>,
    initial_capacities: Vec<usize>,
    failure: Failure,
    restore_fails: bool,
}

impl Decoder for Reusable {
    type Checkpoint = Vec<u32>;

    fn identity(&self) -> &'static str {
        "shared-logits-control/v1"
    }
    fn vocab_size(&self) -> usize {
        257
    }
    fn checkpoint_bytes(&self) -> usize {
        self.tokens.len() * size_of::<u32>()
    }
    fn reset(&mut self) -> Result<()> {
        self.tokens.clear();
        Ok(())
    }
    fn evaluate(&mut self, token: u32) -> Result<Vec<f32>> {
        let mut output = Vec::new();
        self.evaluate_into(token, &mut output)?;
        Ok(output)
    }
    fn evaluate_into(&mut self, token: u32, output: &mut Vec<f32>) -> Result<()> {
        self.initial_capacities.push(output.capacity());
        self.tokens.push(token);
        output.clear();
        output.try_reserve_exact(self.vocab_size())?;
        output.resize(self.vocab_size(), -0.0);
        let selected = usize::try_from(self.tokens.iter().sum::<u32>() % 257)?;
        output[selected] = 1.0;
        match self.failure {
            Failure::None => {}
            Failure::Nonfinite => output[256] = f32::NAN,
            Failure::WrongSize => {
                output.pop();
            }
            Failure::Evaluate => return Err(invalid("injected output failure")),
        }
        Ok(())
    }
    fn checkpoint(&self) -> Result<Vec<u32>> {
        Ok(self.tokens.clone())
    }
    fn restore(&mut self, state: &Vec<u32>) -> Result<()> {
        if self.restore_fails {
            return Err(invalid("injected shared-logits restore failure"));
        }
        self.tokens.clone_from(state);
        Ok(())
    }
}

fn runtime() -> Result<Runtime<Reusable>> {
    Runtime::new(
        Reusable {
            tokens: Vec::new(),
            initial_capacities: Vec::new(),
            failure: Failure::None,
            restore_fails: false,
        },
        Limits {
            context_tokens: 64,
            checkpoints: 4,
            checkpoint_bytes: 64 * 1_024,
        },
    )
}

fn bits(logits: &[f32]) -> Vec<u32> {
    logits.iter().map(|value| value.to_bits()).collect()
}

#[test]
fn shared_logits_capture_restore_and_unique_steps_preserve_storage_and_bits() -> Result<()> {
    let mut runtime = runtime()?;
    runtime.evaluate(1, &|| Ok(()))?;
    let first_pointer = runtime.logits.as_ptr();
    let expected = bits(&runtime.logits);
    let snapshot = runtime.capture(0, &|| Ok(()))?;
    assert!(Arc::ptr_eq(&snapshot.logits, &runtime.logits));
    assert_eq!(snapshot.logits.as_ptr(), first_pointer);
    runtime.checkpoints.push(snapshot);

    runtime.evaluate(2, &|| Ok(()))?;
    assert_ne!(runtime.logits.as_ptr(), first_pointer);
    assert_eq!(runtime.decoder.initial_capacities, [0, 0]);
    assert_eq!(bits(&runtime.checkpoints[0].logits), expected);
    let second_pointer = runtime.logits.as_ptr();
    runtime.evaluate(3, &|| Ok(()))?;
    assert_eq!(runtime.logits.as_ptr(), second_pointer);
    assert_eq!(runtime.decoder.initial_capacities, [0, 0, 257]);

    assert_eq!(runtime.prepare(&[1, 4], &|| Ok(()))?, 1);
    assert_eq!(runtime.logits.as_ptr(), first_pointer);
    assert!(Arc::ptr_eq(&runtime.logits, &runtime.checkpoints[0].logits));
    assert_eq!(bits(&runtime.logits), expected);
    assert_eq!(runtime.decoder.tokens, [1]);
    runtime.ingest_block(&[4], &|| Ok(()))?;
    assert!(runtime.logits.is_empty());
    assert_eq!(bits(&runtime.checkpoints[0].logits), expected);
    runtime.reset()?;
    assert!(runtime.logits.is_empty());
    assert!(runtime.checkpoints.is_empty());
    Ok(())
}

#[test]
fn shared_logits_complete_stop_and_output_caps_retain_immutable_distributions() -> Result<()> {
    let mut runtime = runtime()?;
    let cancellation = Cancellation::default();
    let options = Generation {
        max_tokens: 1,
        stop_tokens: vec![],
    };
    runtime.generate(&[1], &options, &cancellation, |_| Ok(()))?;
    let committed = runtime.tokens().to_vec();
    let old = Arc::clone(&runtime.logits);
    let expected = bits(&old);
    assert!(Arc::ptr_eq(&runtime.checkpoints[0].logits, &old));
    let evaluations = runtime.decoder.initial_capacities.len();
    let stop = greedy(&old)?;
    let complete = runtime.generate(
        &committed,
        &Generation {
            max_tokens: 3,
            stop_tokens: vec![stop],
        },
        &cancellation,
        |_| Err(invalid("stop token must not emit")),
    )?;
    assert_eq!(complete.finish, Finish::Stop);
    assert_eq!(complete.generated_tokens, 0);
    assert_eq!(runtime.decoder.initial_capacities.len(), evaluations);
    assert!(Arc::ptr_eq(&runtime.logits, &old));
    assert!(Arc::ptr_eq(&runtime.checkpoints[1].logits, &old));

    let mut emitted = Vec::new();
    let complete = runtime.generate(
        &committed,
        &Generation {
            max_tokens: 3,
            stop_tokens: vec![],
        },
        &cancellation,
        |token| {
            emitted.push(token);
            Ok(())
        },
    )?;
    assert_eq!(complete.finish, Finish::Length);
    assert_eq!(complete.generated_tokens, 3);
    assert_eq!(emitted.len(), 3);
    assert_eq!(bits(&old), expected);
    assert!(Arc::ptr_eq(&runtime.logits, &runtime.checkpoints[2].logits));
    Ok(())
}

#[test]
fn shared_logits_failed_custom_outputs_delivery_and_cancellation_restore_exact_snapshot()
-> Result<()> {
    for failure in [
        Failure::Nonfinite,
        Failure::WrongSize,
        Failure::Evaluate,
        Failure::None,
    ] {
        for cancel in [false, true] {
            let mut runtime = runtime()?;
            let options = Generation {
                max_tokens: 2,
                stop_tokens: vec![],
            };
            runtime.generate(&[1], &options, &Cancellation::default(), |_| Ok(()))?;
            let committed = runtime.tokens().to_vec();
            let old = Arc::clone(&runtime.logits);
            let expected = bits(&old);
            runtime.decoder.failure = failure;
            let cancellation = Cancellation::default();
            let result = runtime.generate(&committed, &options, &cancellation, |_| {
                if cancel {
                    cancellation.cancel();
                    Ok(())
                } else {
                    Err(invalid("injected shared-logits delivery failure"))
                }
            });
            assert!(result.is_err());
            assert_eq!(runtime.tokens(), committed);
            assert_eq!(runtime.decoder.tokens, committed);
            assert!(Arc::ptr_eq(&runtime.logits, &old));
            assert_eq!(bits(&runtime.logits), expected);
            assert!(!runtime.is_faulted());
            runtime.decoder.failure = Failure::None;
            runtime.generate(&committed, &options, &Cancellation::default(), |_| Ok(()))?;
            assert_eq!(bits(&old), expected);
        }
    }
    Ok(())
}

#[test]
fn shared_logits_failed_restore_keeps_committed_output_and_fault_latch() -> Result<()> {
    let mut runtime = runtime()?;
    let options = Generation {
        max_tokens: 1,
        stop_tokens: vec![],
    };
    runtime.generate(&[1], &options, &Cancellation::default(), |_| Ok(()))?;
    let committed = runtime.tokens().to_vec();
    let old = Arc::clone(&runtime.logits);
    let expected = bits(&old);
    runtime.decoder.restore_fails = true;
    assert!(
        runtime
            .generate(&committed, &options, &Cancellation::default(), |_| {
                Err(invalid("injected shared-logits failed delivery"))
            })
            .is_err()
    );
    assert!(runtime.is_faulted());
    assert_eq!(runtime.tokens(), committed);
    assert!(Arc::ptr_eq(&runtime.logits, &old));
    assert_eq!(bits(&old), expected);
    let evaluations = runtime.decoder.initial_capacities.len();
    assert!(
        runtime
            .generate(&committed, &options, &Cancellation::default(), |_| Ok(()))
            .is_err()
    );
    assert_eq!(runtime.decoder.initial_capacities.len(), evaluations);
    runtime.decoder.restore_fails = false;
    runtime.reset()?;
    assert!(!runtime.is_faulted());
    assert!(runtime.logits.is_empty());
    assert_eq!(bits(&old), expected);
    Ok(())
}

#[test]
fn shared_logits_snapshot_budget_charges_full_capacity_for_every_owner() -> Result<()> {
    let mut runtime = runtime()?;
    runtime.evaluate(1, &|| Ok(()))?;
    // Retained spare output capacity is real storage even though logits use a
    // shorter prefix. Conservatively charge it for each immutable owner.
    mutable_logits(&mut runtime.logits).try_reserve_exact(257)?;
    let snapshot = runtime.capture(0, &|| Ok(()))?;
    let minimum = runtime.logits.capacity() * size_of::<f32>()
        + size_of::<Vec<f32>>()
        + 2 * size_of::<usize>()
        + size_of::<Snapshot<Vec<u32>>>();
    assert!(snapshot.bytes >= minimum);
    runtime.limits.checkpoint_bytes = 2 * snapshot.bytes - 1;
    assert!(runtime.capture(snapshot.bytes, &|| Ok(())).is_err());
    assert!(Arc::ptr_eq(&runtime.logits, &snapshot.logits));
    runtime.limits.checkpoint_bytes += 1;
    let second = runtime.capture(snapshot.bytes, &|| Ok(()))?;
    assert_eq!(second.bytes, snapshot.bytes);
    assert!(Arc::ptr_eq(&snapshot.logits, &second.logits));
    Ok(())
}

#[test]
fn shared_logits_empty_snapshots_keep_reset_output_capacity_unique_and_reusable() -> Result<()> {
    let mut runtime = runtime()?;
    let options = Generation {
        max_tokens: 2,
        stop_tokens: vec![],
    };
    let mut pointer = None;
    for _ in 0..4 {
        runtime.generate(&[1], &options, &Cancellation::default(), |_| Ok(()))?;
        if let Some(expected) = pointer {
            assert_eq!(runtime.logits.as_ptr(), expected);
        }
        pointer = Some(runtime.logits.as_ptr());
        runtime.reset()?;
        assert!(runtime.logits.is_empty());
        assert_eq!(runtime.logits.capacity(), 257);
        let snapshot = runtime.capture(0, &|| Ok(()))?;
        assert!(snapshot.logits.is_empty());
        assert_eq!(snapshot.logits.capacity(), 0);
        assert!(!Arc::ptr_eq(&runtime.logits, &snapshot.logits));
        assert_eq!(Arc::strong_count(&runtime.logits), 1);
    }
    Ok(())
}
