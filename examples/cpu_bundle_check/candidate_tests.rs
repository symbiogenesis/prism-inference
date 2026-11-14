use super::*;

fn arguments() -> Vec<String> {
    let absolute = std::env::temp_dir()
        .join("public-candidate.json")
        .to_string_lossy()
        .into_owned();
    vec![
        absolute.clone(),
        absolute.clone(),
        absolute.clone(),
        absolute,
        "16".into(),
        "1030000".into(),
        "1".repeat(40),
        "k4-v4".into(),
        "1".into(),
        "0".into(),
    ]
}

#[test]
fn source_candidate_selection_requires_explicit_bounded_profile() -> Result<()> {
    let started = Instant::now();
    let now = UNIX_EPOCH + Duration::from_secs(1000);
    let mut input = arguments();
    for cache in ["f16", "k4-v4"] {
        input[7] = cache.into();
        for (tokens, reserve) in [
            (1, 0),
            (8, 67_108_864),
            (16, 134_217_728),
            (32, 268_435_456),
        ] {
            input[8] = tokens.to_string();
            input[9] = reserve.to_string();
            let selected = selection(&input, started, now)?;
            assert_eq!(
                selected.profile.prefill.map_or(1, CpuPrefill::tokens),
                tokens
            );
            assert_eq!(
                selected
                    .profile
                    .prefill
                    .map_or(0, CpuPrefill::scratch_bytes),
                reserve
            );
            assert_eq!(
                selected.profile.cache == KvCachePolicy::Rotated4,
                cache == "k4-v4"
            );
        }
    }
    for (tokens, reserve) in [(1, 1), (8, 0), (4, 1), (32, 268_435_457)] {
        input[8] = tokens.to_string();
        input[9] = reserve.to_string();
        assert!(selection(&input, started, now).is_err());
    }
    input = arguments();
    input[7] = "k4-v3".into();
    assert!(selection(&input, started, now).is_err());
    assert!(selection(&input[..7], started, now).is_err());
    assert!(selection(&input[..9], started, now).is_err());
    input.push("extra".into());
    assert!(selection(&input, started, now).is_err());
    input = arguments();
    input[4] = "33".into();
    assert!(selection(&input, started, now).is_err());
    input = arguments();
    input[5] = "1240001".into();
    assert!(selection(&input, started, now).is_err());
    input = arguments();
    input[6] = "A".repeat(40);
    assert!(selection(&input, started, now).is_err());
    Ok(())
}

#[test]
fn source_candidate_receipts_separate_profile_and_unknown_observations() -> Result<()> {
    let mut input = arguments();
    input[8] = "16".into();
    input[9] = "134217728".into();
    let selected = selection(
        &input,
        Instant::now(),
        UNIX_EPOCH + Duration::from_secs(1000),
    )?;
    let receipt = record(&selected, &Observation::default(), false, Duration::ZERO)?;
    assert_eq!(receipt["schema_version"], 2);
    assert_eq!(receipt["kind"], "swift-cpu-source-candidate-arm");
    assert_eq!(receipt["kv_cache"], "k4-v4");
    assert_eq!(receipt["context_tokens"], 8192);
    assert_eq!(receipt["cpu_prefill_tokens"], 16);
    assert_eq!(receipt["cpu_prefill_scratch_bytes"], 134_217_728);
    assert_eq!(receipt["execution_device"], "cpu");
    assert_eq!(receipt["arm_succeeded"], false);
    for field in [
        "executable_sha256",
        "effective_row_workers",
        "output_text",
        "request_milliseconds",
    ] {
        assert!(receipt[field].is_null());
    }
    Ok(())
}

#[test]
fn source_candidate_failure_keeps_partial_output_and_distinct_create_new_receipts() -> Result<()> {
    let directory = std::env::temp_dir().join(format!(
        "cpu-candidate-{}-{}",
        std::process::id(),
        SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos()
    ));
    fs::create_dir(&directory)?;
    let mut selected = selection(
        &arguments(),
        Instant::now(),
        UNIX_EPOCH + Duration::from_secs(1000),
    )?;
    selected.evidence.clone_from(&directory);
    let started = Instant::now();
    assert!(
        capture(&selected, started, |_, observed| {
            observed.stage = "injected-failure";
            observed.request_milliseconds = Some(1.0);
            observed.output.text = "partial".into();
            Err(reject("injected failure"))
        })
        .is_err()
    );
    let final_path = directory.join("cpu-candidate.json");
    let original = fs::read(&final_path)?;
    let receipt: Value = serde_json::from_slice(&original)?;
    assert_eq!(receipt["kind"], "swift-cpu-source-candidate-arm");
    assert_eq!(receipt["output_text"], "partial");
    assert_eq!(receipt["output_committed"], false);
    assert!(!directory.join("cpu-bundle.json").exists());
    assert!(capture(&selected, started, |_, _| Err(reject("must not run"))).is_err());
    assert_eq!(fs::read(&final_path)?, original);
    fs::remove_file(final_path)?;
    fs::remove_file(directory.join("cpu-candidate-started.json"))?;
    fs::remove_dir(directory)?;
    Ok(())
}
