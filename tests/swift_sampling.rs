//! Swift production sampling participates in the runtime's whole-request rollback.
#![forbid(unsafe_code)]

use bonsai_prism_runtime::{
    Decoder, Result,
    runtime::{Cancellation, Finish, Generation, Limits, Prefill, Runtime},
    sampling::{Sampler, SwiftSampler},
};

struct FullVocabulary {
    tokens: Vec<u32>,
    peak: Option<usize>,
}

impl Decoder for FullVocabulary {
    type Checkpoint = Vec<u32>;
    fn identity(&self) -> &'static str {
        "synthetic-full-swift-vocabulary"
    }
    fn vocab_size(&self) -> usize {
        SwiftSampler::VOCABULARY
    }
    fn checkpoint_bytes(&self) -> usize {
        self.tokens.len() * 4
    }
    fn reset(&mut self) -> Result<()> {
        self.tokens.clear();
        Ok(())
    }
    fn evaluate(&mut self, token: u32) -> Result<Vec<f32>> {
        self.tokens.push(token);
        let mut logits = vec![0.0; self.vocab_size()];
        if let Some(peak) = self.peak {
            logits[peak] = 1000.0;
        }
        Ok(logits)
    }
    fn checkpoint(&self) -> Result<Self::Checkpoint> {
        Ok(self.tokens.clone())
    }
    fn restore(&mut self, checkpoint: &Self::Checkpoint) -> Result<()> {
        self.tokens.clone_from(checkpoint);
        Ok(())
    }
}

fn runtime(peak: Option<usize>) -> Result<Runtime<FullVocabulary>> {
    Runtime::new(
        FullVocabulary {
            tokens: Vec::new(),
            peak,
        },
        Limits {
            context_tokens: 8192,
            checkpoints: 0,
            checkpoint_bytes: 2 * 1024 * 1024,
        },
    )
}

#[test]
fn swift_rng_rolls_back_after_cancelled_or_abandoned_stream_then_retries() -> Result<()> {
    for cancel in [false, true] {
        let mut runtime = runtime(None)?;
        let mut sampler = SwiftSampler::new(4242)?;
        let mut control = sampler.clone();
        let options = Generation {
            max_tokens: 3,
            stop_tokens: vec![],
        };
        let cancellation = Cancellation::default();
        let mut emitted = Vec::new();
        assert!(
            runtime
                .generate_sampled(
                    &[1, 2],
                    &options,
                    Prefill::default(),
                    &cancellation,
                    &mut sampler,
                    |token| {
                        emitted.push(token);
                        if cancel {
                            cancellation.cancel();
                            Ok(())
                        } else {
                            Err(std::io::Error::other("abandoned output").into())
                        }
                    }
                )
                .is_err()
        );
        assert_eq!(emitted.len(), 1);
        assert_eq!(runtime.tokens(), [0_u32; 0]);
        assert_eq!(runtime.decoder().tokens, [0_u32; 0]);
        assert!(!runtime.is_faulted());
        let mut retried = Vec::new();
        let completion = runtime.generate_sampled(
            &[1, 2],
            &options,
            Prefill::default(),
            &Cancellation::default(),
            &mut sampler,
            |token| {
                retried.push(token);
                Ok(())
            },
        )?;
        assert_eq!(completion.finish, Finish::Length);
        assert_eq!(completion.generated_tokens, 3);
        let logits = vec![0.0; SwiftSampler::VOCABULARY];
        let expected = (0..3)
            .map(|_| control.sample(&logits))
            .collect::<Result<Vec<_>>>()?;
        assert_eq!(retried, expected);
        assert_eq!(emitted[0], retried[0]);
        assert_eq!(sampler.sample(&logits)?, control.sample(&logits)?);
    }
    Ok(())
}

#[test]
fn swift_declared_stop_is_not_delivered_and_commits_its_draw() -> Result<()> {
    // The pinned Swift metadata declares EOS248046. The graph is synthetic;
    // verified declaration extraction is tested separately by SpecialTokens.
    let mut runtime = runtime(Some(248_046))?;
    let mut sampler = SwiftSampler::new(0)?;
    let mut control = sampler.clone();
    let mut logits = vec![0.0; SwiftSampler::VOCABULARY];
    logits[248_046] = 1000.0;
    assert_eq!(control.sample(&logits)?, 248_046);
    let completion = runtime.generate_sampled(
        &[1],
        &Generation {
            max_tokens: 4096,
            stop_tokens: vec![248_046],
        },
        Prefill::default(),
        &Cancellation::default(),
        &mut sampler,
        |_| Err(std::io::Error::other("stop must not be delivered").into()),
    )?;
    assert_eq!(completion.finish, Finish::Stop);
    assert_eq!(completion.generated_tokens, 0);
    assert_eq!(runtime.tokens(), &[1]);
    logits.fill(0.0);
    assert_eq!(sampler.sample(&logits)?, control.sample(&logits)?);
    Ok(())
}

#[test]
fn swift_rejects_invalid_full_vocabulary_without_consuming_draws_and_resets_seed() -> Result<()> {
    assert!(SwiftSampler::new(u32::MAX).is_err());
    let mut sampler = SwiftSampler::new(1)?;
    let mut control = sampler.clone();
    for logits in [&[][..], &[0.0], &vec![0.0; 248_319], &vec![0.0; 248_321]] {
        assert!(sampler.sample(logits).is_err());
    }
    let mut logits = vec![0.0; SwiftSampler::VOCABULARY];
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        logits[248_319] = value;
        assert!(sampler.sample(&logits).is_err());
    }
    logits[248_319] = 0.0;
    for _ in 0..4 {
        assert_eq!(sampler.sample(&logits)?, control.sample(&logits)?);
    }
    sampler.reset();
    control = SwiftSampler::new(1)?;
    for _ in 0..4 {
        assert_eq!(sampler.sample(&logits)?, control.sample(&logits)?);
    }
    Ok(())
}
