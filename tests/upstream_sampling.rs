//! Focused native RNG parity and stateful candidate storage contracts.
#![forbid(unsafe_code)]

use bonsai_prism_runtime::sampling::upstream::{Candidate, Candidates, Config, State};
use bonsai_prism_runtime::sampling::{Sampler, SwiftSampler};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::fmt::Write as _;

type TestResult = Result<(), Box<dyn std::error::Error + Send + Sync>>;

fn candidates(count: usize) -> Result<Candidates, Box<dyn std::error::Error + Send + Sync>> {
    Candidates::new(
        (0..count)
            .map(|index| {
                Ok(Candidate {
                    id: i32::try_from(index)?,
                    logit: 0.0,
                    probability: 0.0,
                })
            })
            .collect::<Result<Vec<_>, Box<dyn std::error::Error + Send + Sync>>>()?,
        -1,
        false,
    )
}

#[test]
fn production_candidates_admit_swift_before_filtering_and_reject_the_bound() -> TestResult {
    let mut swift = candidates(248_320)?;
    assert_eq!(swift.backing_len(), 248_320);
    State::new(Config::TopK(20))?.apply(&mut swift)?;
    assert_eq!(swift.data().len(), 20);
    assert_eq!(swift.backing_len(), 248_320);
    assert!(candidates(1_048_576).is_ok());
    assert!(candidates(1_048_577).is_err());
    assert!(
        State::new(Config::Penalties {
            vocabulary_size: 248_320,
            last_n: 64,
            repeat: 1.0,
            frequency: 0.0,
            presence: 0.0,
        })
        .is_ok()
    );
    Ok(())
}

fn swift_logits(kind: &str) -> Result<Vec<f32>, Box<dyn std::error::Error + Send + Sync>> {
    (0..248_320)
        .map(|index| {
            Ok(match kind {
                "uniform" => 0.0,
                "permuted" => {
                    let value = index * 73 % 248_320;
                    f32::from(u16::try_from(value / 1024)?)
                        + f32::from(u16::try_from(value % 1024)?) / 1024.0
                        - 128.0
                }
                "tied-boundary" => f32::from(u16::try_from(index * 17 % 7)?) - 3.0,
                "single" => {
                    if index == 248_319 {
                        1000.0
                    } else {
                        -1000.0
                    }
                }
                _ => return Err("unknown native Swift sampler input".into()),
            })
        })
        .collect()
}

fn assert_native_candidates(data: &Candidates, expected: &[Value]) -> TestResult {
    assert_eq!(data.data().len(), expected.len());
    for (candidate, native) in data.data().iter().zip(expected) {
        assert_eq!(
            i64::from(candidate.id),
            native["Id"].as_i64().ok_or("native id")?
        );
        assert_eq!(
            u64::from(candidate.logit.to_bits()),
            native["LogitBits"].as_u64().ok_or("native logit")?
        );
        let probability = f32::from_bits(u32::try_from(
            native["ProbabilityBits"]
                .as_u64()
                .ok_or("native probability")?,
        )?);
        assert!((candidate.probability - probability).abs() <= 1e-6);
    }
    Ok(())
}

#[test]
fn swift_chain_matches_verified_pinned_arm64_candidates_and_seeded_draws() -> TestResult {
    // Canonical LF bytes retain their capture digest on CRLF Git checkouts.
    let bytes = include_str!("fixtures/swift-sampler.json").replace("\r\n", "\n");
    let mut digest = String::new();
    for byte in Sha256::digest(bytes.as_bytes()) {
        write!(&mut digest, "{byte:02x}")?;
    }
    assert_eq!(
        digest,
        "662e81fdd618b304b2bcc4933dc0fa6ec6a7808251253437b1d2a59bf3404242"
    );
    let fixture: Value = serde_json::from_str(&bytes)?;
    assert_eq!(
        fixture["RuntimeCommit"],
        "73bfe6cc485554822f80c473a3c0c4f25e80f41b"
    );
    assert_eq!(
        fixture["ArchiveSha256"],
        "cc808b33d35c72a3bd5c69a0d609d422426feb9dce23868e77992b6c33b8ddb3"
    );
    for case in fixture["Cases"].as_array().ok_or("missing native cases")? {
        let logits = swift_logits(case["Kind"].as_str().ok_or("missing native input kind")?)?;
        let mut input = Vec::new();
        for logit in &logits {
            input.extend_from_slice(&logit.to_le_bytes());
        }
        let mut input_digest = String::new();
        for byte in Sha256::digest(&input) {
            write!(&mut input_digest, "{byte:02x}")?;
        }
        assert_eq!(input_digest, case["LogitsSha256"]);
        let seed = u32::try_from(case["Seed"].as_u64().ok_or("missing seed")?)?;
        let mut distribution = State::new(Config::Distribution(seed))?;
        let mut sampler = SwiftSampler::new(seed)?;
        for (draw, selected) in case["Selected"]
            .as_array()
            .ok_or("missing draws")?
            .iter()
            .enumerate()
        {
            let mut data = Candidates::new(
                logits
                    .iter()
                    .enumerate()
                    .map(|(index, &logit)| {
                        Ok(Candidate {
                            id: i32::try_from(index)?,
                            logit,
                            probability: 0.0,
                        })
                    })
                    .collect::<Result<Vec<_>, Box<dyn std::error::Error + Send + Sync>>>()?,
                -1,
                false,
            )?;
            for config in [
                Config::TopK(20),
                Config::TopP {
                    probability: 0.95,
                    min_keep: 1,
                },
                Config::MinP {
                    probability: 0.0,
                    min_keep: 1,
                },
                Config::Temperature(1.0),
            ] {
                State::new(config)?.apply(&mut data)?;
            }
            distribution.apply(&mut data)?;
            let index = usize::try_from(data.selected())?;
            assert_eq!(
                i64::from(data.data()[index].id),
                selected.as_i64().ok_or("invalid native selection")?,
                "kind={} seed={seed} draw={draw}",
                case["Kind"]
            );
            assert_eq!(
                u64::from(sampler.sample(&logits)?),
                selected.as_u64().ok_or("native selected token")?
            );
            if draw == 0 {
                let expected = case["FirstCandidates"]
                    .as_array()
                    .ok_or("missing candidates")?;
                assert_native_candidates(&data, expected)?;
            }
        }
    }
    Ok(())
}

#[test]
fn distribution_and_xtc_match_real_native_msvc_draws_and_reset() -> TestResult {
    let bytes = include_bytes!("fixtures/sampler-rng.json");
    let mut digest = String::with_capacity(64);
    for byte in Sha256::digest(bytes) {
        write!(&mut digest, "{byte:02x}")?;
    }
    assert_eq!(
        digest,
        "5e728e604357e474898180cfb866bdba62102d95048ff8bf79a8fcea12938310"
    );
    let fixture: Value = serde_json::from_slice(bytes)?;
    assert_eq!(
        fixture["UpstreamCommit"],
        "b87ca7225456b06b0975b8239c66b5aecf07e37c"
    );
    let mut distribution = State::new(Config::Distribution(4242))?;
    let mut xtc = State::new(Config::Xtc {
        probability: 0.5,
        threshold: 0.1,
        min_keep: 1,
        seed: 12345,
    })?;
    for _ in 0..2 {
        for (count, selected) in fixture["DistributionCounts"]
            .as_array()
            .ok_or("missing counts")?
            .iter()
            .zip(
                fixture["DistributionSelected"]
                    .as_array()
                    .ok_or("missing selections")?,
            )
        {
            let mut buffer =
                candidates(usize::try_from(count.as_u64().ok_or("count must be u64")?)?)?;
            distribution.apply(&mut buffer)?;
            assert_eq!(
                buffer.selected(),
                selected.as_i64().ok_or("selected must be i64")?
            );
            assert!(!buffer.sorted());
        }
        for count in fixture["XtcCounts"].as_array().ok_or("missing XTC sizes")? {
            let mut buffer = candidates(5)?;
            xtc.apply(&mut buffer)?;
            assert_eq!(
                buffer.data().len(),
                usize::try_from(count.as_u64().ok_or("count must be u64")?)?
            );
        }
        distribution.reset();
        xtc.reset();
    }
    Ok(())
}

#[test]
fn cloned_views_retain_offset_backing_and_independent_mutations() -> TestResult {
    let mut original = candidates(5)?;
    let mut xtc = State::new(Config::Xtc {
        probability: 1.0,
        threshold: 0.1,
        min_keep: 1,
        seed: 0,
    })?;
    xtc.apply(&mut original)?;
    assert_eq!(original.offset(), 4);
    assert_eq!(original.backing_len(), 5);
    assert_eq!(original.data().len(), 1);
    assert!(original.sorted());
    let mut copied = original.clone();
    State::new(Config::Distribution(4242))?.apply(&mut copied)?;
    assert_eq!(copied.offset(), 4);
    assert_eq!(copied.selected(), 0);
    assert_eq!(copied.data()[0].probability.to_bits(), 1.0_f32.to_bits());
    assert_eq!(original.selected(), -1);
    assert_eq!(original.data()[0].probability.to_bits(), 0.2_f32.to_bits());
    let mut sampler = State::new(Config::Distribution(4242))?;
    let mut clone = sampler.clone();
    for _ in 0..5 {
        let mut left = candidates(257)?;
        let mut right = left.clone();
        sampler.apply(&mut left)?;
        clone.apply(&mut right)?;
        assert_eq!(left.selected(), right.selected());
        assert_eq!(left.data(), right.data());
    }
    Ok(())
}

#[test]
fn history_eviction_clone_reset_and_zero_temperature_masks_are_owned() -> TestResult {
    let mut sampler = State::new(Config::Penalties {
        vocabulary_size: 3,
        last_n: 2,
        repeat: 2.0,
        frequency: 0.5,
        presence: 0.25,
    })?;
    for token in [1, 1, 2] {
        sampler.accept(token);
    }
    let mut copied = sampler.clone();
    sampler.reset();
    let mut data = Candidates::new(
        vec![
            Candidate {
                id: 1,
                logit: 2.0,
                probability: 0.0,
            },
            Candidate {
                id: 2,
                logit: -2.0,
                probability: 0.0,
            },
        ],
        -1,
        true,
    )?;
    copied.apply(&mut data)?;
    assert_eq!(data.data()[0].logit.to_bits(), 0.25_f32.to_bits());
    assert_eq!(data.data()[1].logit.to_bits(), (-4.75_f32).to_bits());
    assert!(!data.sorted());
    let unchanged = data.clone();
    sampler.apply(&mut data)?;
    assert_eq!(data.data(), unchanged.data());
    let mut ties = Candidates::new(
        vec![
            Candidate {
                id: 0,
                logit: 1.0,
                probability: 0.2,
            },
            Candidate {
                id: 1,
                logit: 2.0,
                probability: 0.3,
            },
            Candidate {
                id: 2,
                logit: 2.0,
                probability: 0.5,
            },
        ],
        -1,
        false,
    )?;
    State::new(Config::Temperature(0.0))?.apply(&mut ties)?;
    assert_eq!(
        ties.data()
            .iter()
            .map(|entry| entry.logit.to_bits())
            .collect::<Vec<_>>(),
        [
            f32::NEG_INFINITY.to_bits(),
            2.0_f32.to_bits(),
            f32::NEG_INFINITY.to_bits()
        ]
    );
    assert!(State::new(Config::Temperature(f32::NAN)).is_err());
    assert!(Candidates::new(vec![], -1, false).is_err());
    assert!(
        Candidates::new(
            vec![Candidate {
                id: 0,
                logit: 0.0,
                probability: 0.0
            }],
            1,
            false
        )
        .is_err()
    );
    Ok(())
}
