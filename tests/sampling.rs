#![forbid(unsafe_code)]
//! Seeded reference-sampler ranking, probability and admission regressions.

use bonsai_prism_runtime::{
    Result,
    runtime::greedy,
    sampling::{Greedy, ReferenceSampler, Sampler, Sampling},
};

const UNFILTERED: Sampling = Sampling {
    temperature: 1.0,
    top_k: 0,
    top_p: 1.0,
};

fn draw(options: Sampling, logits: &[f32]) -> Result<u32> {
    ReferenceSampler::new(options, 0)?.sample(logits)
}

#[test]
fn default_options_match_explicit_reference_settings() {
    let options = Sampling::default();
    assert_eq!(options.temperature.to_bits(), 1.0_f32.to_bits());
    assert_eq!(options.top_k, 20);
    assert_eq!(options.top_p.to_bits(), 0.95_f32.to_bits());
}

#[test]
fn greedy_and_adapter_validate_every_logit_and_preserve_first_ties() -> Result<()> {
    let mut sampler = Greedy;
    for (logits, expected) in [
        (&[1.0, 3.0, 3.0][..], 1),
        (&[-0.0, 0.0, -0.0][..], 0),
        (&[-3.0, -1.0, -1.0, -2.0][..], 1),
        (&[-2.0, -1.0, 0.0, 1.0][..], 3),
    ] {
        assert_eq!(greedy(logits)?, expected);
        assert_eq!(sampler.sample(logits)?, expected);
    }
    assert!(greedy(&[]).is_err());
    assert!(sampler.sample(&[]).is_err());
    for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(greedy(&[invalid]).is_err());
        assert!(sampler.sample(&[invalid]).is_err());
        for position in 0..4 {
            let mut logits = [f32::MAX, -1.0, -2.0, -3.0];
            logits[position] = invalid;
            assert!(greedy(&logits).is_err());
            assert!(sampler.sample(&logits).is_err());
        }
    }
    Ok(())
}

#[test]
fn splitmix64_high_word_has_a_pinned_categorical_sequence() -> Result<()> {
    let mut sampler = ReferenceSampler::new(UNFILTERED, 0)?;
    // Uniform eight-way selection reads the three high bits of each SplitMix64
    // word. Seed zero starts e220a8397b1dcdaf, 6e789e6aa1b965f4,
    // 06c45d188009454f, f88bb8a8724c81ec, 1b39896a51a8749b,
    // 53cb9f0c747ea2ea, 2c829abe1f4532e1, c584133ac916ab3c.
    for expected in [7, 3, 0, 7, 0, 2, 1, 6] {
        assert_eq!(sampler.sample(&[0.0; 8])?, expected);
    }
    Ok(())
}

#[test]
fn clones_preserve_midstream_rng_state_and_seed_changes_the_draw() -> Result<()> {
    let mut original = ReferenceSampler::new(UNFILTERED, 0)?;
    original.sample(&[0.0; 8])?;
    let mut cloned = original.clone();
    for _ in 0..32 {
        assert_eq!(original.sample(&[0.0; 8])?, cloned.sample(&[0.0; 8])?);
    }
    let mut different = ReferenceSampler::new(UNFILTERED, 1)?;
    assert_eq!(different.sample(&[0.0; 8])?, 4);
    Ok(())
}

#[test]
fn ranked_top_k_breaks_equal_logits_by_ascending_token_id() -> Result<()> {
    let options = Sampling {
        top_k: 2,
        ..UNFILTERED
    };
    let mut sampler = ReferenceSampler::new(options, 0)?;
    for expected in [2, 1, 1, 2, 1, 1, 1, 2] {
        assert_eq!(sampler.sample(&[1.0, 3.0, 3.0, 3.0])?, expected);
    }
    for logits in [&[-0.0, 0.0][..], &[0.0, -0.0]] {
        assert_eq!(
            draw(
                Sampling {
                    top_k: 1,
                    ..UNFILTERED
                },
                logits,
            )?,
            0
        );
    }
    Ok(())
}

#[test]
fn bounded_top_k_matches_a_full_sort_across_rank_and_probability_boundaries() -> Result<()> {
    let logits: Vec<f32> = (0..257)
        .map(|index| match (index * 73 % 257) % 7 {
            0 => -0.0,
            1 => 0.0,
            2 => -3.0,
            3 => 2.0,
            4 => 1.0,
            5 => -1.0,
            _ => 3.0,
        })
        .collect();
    let mut ranked: Vec<_> = logits.iter().enumerate().collect();
    ranked.sort_unstable_by(|(left_token, left), (right_token, right)| {
        right
            .partial_cmp(left)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left_token.cmp(right_token))
    });
    for top_k in [1, 2, 20, 37, 128, 256, 257, 258] {
        let prefix = &ranked[..top_k.min(ranked.len())];
        let prefix_logits: Vec<_> = prefix.iter().map(|(_, logit)| **logit).collect();
        for temperature in [0.25, 1.0, 4.0] {
            for top_p in [f32::from_bits(1), 0.5, 0.95, 1.0] {
                let options = Sampling {
                    temperature,
                    top_k,
                    top_p,
                };
                let mut sampler = ReferenceSampler::new(options, u64::MAX)?;
                let mut reference = ReferenceSampler::new(
                    Sampling {
                        top_k: 0,
                        ..options
                    },
                    u64::MAX,
                )?;
                for _ in 0..32 {
                    let selected = usize::try_from(reference.sample(&prefix_logits)?)?;
                    assert_eq!(
                        usize::try_from(sampler.sample(&logits)?)?,
                        prefix[selected].0,
                        "top_k={top_k}, temperature={temperature}, top_p={top_p}"
                    );
                }
            }
        }
    }
    Ok(())
}

#[test]
fn bounded_top_k_validates_discarded_logits_without_consuming_a_draw() -> Result<()> {
    let options = Sampling {
        top_k: 2,
        ..UNFILTERED
    };
    let mut sampler = ReferenceSampler::new(options, 0)?;
    let mut control = sampler.clone();
    for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut logits = [0.0; 64];
        logits[0] = 10.0;
        logits[63] = invalid;
        assert!(sampler.sample(&logits).is_err());
        assert_eq!(sampler.sample(&[0.0; 64])?, control.sample(&[0.0; 64])?);
    }
    Ok(())
}

#[test]
fn top_p_retains_the_crossing_token_and_renormalizes_its_prefix() -> Result<()> {
    let logits = [0.0; 4];
    assert_eq!(
        draw(
            Sampling {
                top_p: 0.26,
                ..UNFILTERED
            },
            &logits,
        )?,
        1
    );
    assert_eq!(
        draw(
            Sampling {
                top_p: 0.5,
                ..UNFILTERED
            },
            &logits,
        )?,
        1
    );
    assert_eq!(
        draw(
            Sampling {
                top_p: f32::from_bits(0.5_f32.to_bits() + 1),
                ..UNFILTERED
            },
            &logits,
        )?,
        2
    );
    assert_eq!(
        draw(
            Sampling {
                top_p: f32::from_bits(1),
                ..UNFILTERED
            },
            &logits,
        )?,
        0
    );
    Ok(())
}

#[test]
fn top_k_is_applied_before_the_top_p_probability_threshold() -> Result<()> {
    assert_eq!(
        draw(
            Sampling {
                top_k: 2,
                top_p: 0.5,
                ..UNFILTERED
            },
            &[0.0; 4],
        )?,
        0
    );
    assert_eq!(
        draw(
            Sampling {
                top_k: 1_048_576,
                ..UNFILTERED
            },
            &[0.0; 8],
        )?,
        7
    );
    Ok(())
}

#[test]
fn temperature_changes_softmax_before_the_nucleus_is_selected() -> Result<()> {
    let logits = [2.0, 1.0, 0.0];
    assert_eq!(
        draw(
            Sampling {
                temperature: 0.1,
                top_p: 0.8,
                ..UNFILTERED
            },
            &logits,
        )?,
        0
    );
    assert_eq!(
        draw(
            Sampling {
                temperature: 10.0,
                top_p: 0.8,
                ..UNFILTERED
            },
            &logits,
        )?,
        2
    );
    assert_eq!(draw(UNFILTERED, &logits)?, 1);
    assert_eq!(draw(UNFILTERED, &[0.0, 2.0, 1.0])?, 2);
    Ok(())
}

#[test]
fn stable_softmax_accepts_extreme_finite_logits_and_temperature() -> Result<()> {
    assert_eq!(
        draw(
            Sampling {
                temperature: f32::from_bits(1),
                ..UNFILTERED
            },
            &[-f32::MAX, f32::MAX, 0.0],
        )?,
        1
    );
    assert_eq!(
        draw(
            Sampling {
                temperature: f32::MAX,
                ..UNFILTERED
            },
            &[-f32::MAX, f32::MAX],
        )?,
        0
    );
    assert_eq!(draw(UNFILTERED, &[f32::MAX; 8])?, 7);
    assert_eq!(draw(UNFILTERED, &[-f32::MAX; 8])?, 7);
    Ok(())
}

#[test]
fn invalid_sampling_options_are_rejected_at_construction() {
    for temperature in [0.0, -0.0, -1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(
            ReferenceSampler::new(
                Sampling {
                    temperature,
                    ..UNFILTERED
                },
                0,
            )
            .is_err()
        );
    }
    for top_p in [
        0.0,
        -0.0,
        -1.0,
        1.01,
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
    ] {
        assert!(
            ReferenceSampler::new(
                Sampling {
                    top_p,
                    ..UNFILTERED
                },
                0,
            )
            .is_err()
        );
    }
    for top_k in [1_048_577, usize::MAX] {
        assert!(
            ReferenceSampler::new(
                Sampling {
                    top_k,
                    ..UNFILTERED
                },
                0,
            )
            .is_err()
        );
    }
}

#[test]
fn invalid_logits_do_not_advance_the_generator() -> Result<()> {
    let mut sampler = ReferenceSampler::new(UNFILTERED, 0)?;
    let mut untouched = sampler.clone();
    for logits in [
        &[][..],
        &[1.0, f32::NAN],
        &[f32::INFINITY, 1.0],
        &[1.0, f32::NEG_INFINITY],
    ] {
        assert!(sampler.sample(logits).is_err());
        assert_eq!(sampler.sample(&[0.0; 8])?, untouched.sample(&[0.0; 8])?);
    }
    assert!(sampler.sample(&vec![0.0; 1_048_577]).is_err());
    for _ in 0..8 {
        assert_eq!(sampler.sample(&[0.0; 8])?, untouched.sample(&[0.0; 8])?);
    }
    Ok(())
}

#[test]
fn vocabulary_admits_its_maximum_and_keeps_token_ids_in_bounds() -> Result<()> {
    let mut logits = vec![0.0; 1_048_576];
    logits[1_048_575] = 1.0;
    assert_eq!(
        draw(
            Sampling {
                top_k: 1,
                ..UNFILTERED
            },
            &logits,
        )?,
        1_048_575
    );
    Ok(())
}

#[test]
fn successful_single_candidate_calls_consume_exactly_one_draw() -> Result<()> {
    let mut sampler = ReferenceSampler::new(UNFILTERED, u64::MAX)?;
    let mut control = sampler.clone();
    assert_eq!(sampler.sample(&[7.0])?, 0);
    control.sample(&[0.0; 8])?;
    for _ in 0..8 {
        assert_eq!(sampler.sample(&[0.0; 8])?, control.sample(&[0.0; 8])?);
    }
    Ok(())
}
