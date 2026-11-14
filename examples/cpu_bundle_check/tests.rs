use super::*;

fn bounded_selection() -> Selection {
    let generation_deadline = Instant::now() + Duration::from_secs(28);
    let deadline = generation_deadline + RECEIPT_RESERVE;
    Selection {
        model: PathBuf::from("model"),
        adapter: PathBuf::from("adapter"),
        evidence: PathBuf::from("evidence"),
        messages: PathBuf::from("messages"),
        output_tokens: 16,
        source_commit: "0".repeat(40),
        profile: Profile,
        deadline_unix_milliseconds: 1,
        deadline,
        generation_deadline,
    }
}

fn complete_observation() -> Observation {
    let completion = Completion {
        prompt_tokens: 7,
        reused_tokens: 0,
        generated_tokens: 2,
        finish: Finish::Length,
    };
    let mut observation = Observation {
        stage: "complete",
        prompt_tokens: Some(7),
        request_milliseconds: Some(3.0),
        completion: Some(completion),
        output: CapturedOutput {
            text: "oak 🌳".to_owned(),
            terminal: Some(completion),
            first_text_delta_milliseconds: Some(1.0),
        },
        ..Observation::default()
    };
    observation.timings.entered = true;
    observation.timings.succeeded = true;
    observation.timings.prompt_tokens = 7;
    observation.timings.reused_tokens = Some(0);
    observation.timings.committed_generated_tokens = Some(2);
    observation
}

#[test]
fn deadlines_require_original_bounded_absolute_time() -> Result<()> {
    let started = Instant::now();
    let now = UNIX_EPOCH + Duration::from_secs(1000);
    for invalid in ["", "expired", "999999", "1000000", "1004000", "1240001"] {
        assert!(bounded_deadline(invalid, started, now).is_err());
    }
    let (milliseconds, deadline) = bounded_deadline("1030000", started, now)?;
    assert_eq!(milliseconds, 1_030_000);
    assert_eq!(deadline.duration_since(started), Duration::from_secs(30));
    Ok(())
}

#[test]
fn selection_rejects_relative_paths_bad_commit_and_output_overflow() {
    let input_path = std::env::temp_dir().join("cpu-bundle-public.json");
    let absolute = input_path.to_string_lossy().into_owned();
    let mut arguments = vec![
        absolute.clone(),
        absolute.clone(),
        absolute.clone(),
        absolute,
        "16".to_owned(),
        "1030000".to_owned(),
        LEGACY_F16_SOURCE.to_owned(),
    ];
    let started = Instant::now();
    let now = UNIX_EPOCH + Duration::from_secs(1000);
    assert!(selection(&arguments, started, now).is_ok());
    arguments[0] = "relative-model".to_owned();
    assert!(selection(&arguments, started, now).is_err());
    arguments[0] = input_path.to_string_lossy().into_owned();
    arguments[4] = "33".to_owned();
    assert!(selection(&arguments, started, now).is_err());
    arguments[4] = "16".to_owned();
    arguments[6] = "A".repeat(40);
    assert!(selection(&arguments, started, now).is_err());
    if cfg!(cpu_bundle_legacy_f16) {
        arguments[6] = "1".repeat(40);
        assert!(selection(&arguments, started, now).is_err());
    }
}

#[test]
fn frozen_selection_rejects_candidate_suffix_and_preserves_schema_one() -> Result<()> {
    assert!(parse_profile(&["k4-v4".into(), "1".into(), "0".into()]).is_err());
    let identity = identity_record(&bounded_selection(), &Observation::default());
    assert_eq!(identity["schema_version"], 1);
    assert_eq!(identity["kind"], "swift-cpu-bundle-arm");
    assert_eq!(identity["kv_cache"], "f16");
    assert!(identity.get("cpu_prefill_tokens").is_none());
    assert_eq!(receipt_names(), ("cpu-bundle-started.json", "cpu-bundle.json"));
    Ok(())
}

#[test]
fn output_rejects_overflow_and_any_delivery_after_terminal() -> Result<()> {
    let mut output = CapturedOutput::default();
    output.push(&TextEvent::Delta("oak"))?;
    assert!(
        output
            .push(&TextEvent::Delta(&"x".repeat(MAX_OUTPUT_BYTES)))
            .is_err()
    );
    assert_eq!(output.text, "oak");
    let completion = Completion {
        prompt_tokens: 7,
        reused_tokens: 0,
        generated_tokens: 2,
        finish: Finish::Length,
    };
    output.push(&TextEvent::Complete(completion))?;
    assert!(output.push(&TextEvent::Complete(completion)).is_err());
    assert!(output.push(&TextEvent::Delta("late")).is_err());
    assert_eq!(output.text, "oak");
    Ok(())
}

#[test]
fn first_text_event_timing_ignores_empty_failed_and_subsequent_deliveries() -> Result<()> {
    let mut output = CapturedOutput::default();
    output.push_observed(&TextEvent::Delta(""), Duration::from_millis(1))?;
    assert!(output.first_text_delta_milliseconds.is_none());
    assert!(
        output
            .push_observed(
                &TextEvent::Delta(&"x".repeat(MAX_OUTPUT_BYTES + 1)),
                Duration::from_millis(2),
            )
            .is_err()
    );
    assert!(output.first_text_delta_milliseconds.is_none());
    output.push_observed(&TextEvent::Delta("oak"), Duration::from_millis(3))?;
    let first = output.first_text_delta_milliseconds;
    output.push_observed(&TextEvent::Delta(" tree"), Duration::from_millis(4))?;
    assert_eq!(
        output.first_text_delta_milliseconds.map(f64::to_bits),
        first.map(f64::to_bits),
    );
    assert_eq!(output.text, "oak tree");
    let receipt = record(
        &bounded_selection(),
        &Observation {
            output,
            request_milliseconds: Some(5.0),
            ..Observation::default()
        },
        false,
        Duration::from_millis(6),
    )?;
    assert!(
        receipt["first_text_delta_milliseconds"]
            .as_f64()
            .is_some_and(|value| (value - 3.0).abs() < f64::EPSILON)
    );
    assert_eq!(receipt["generated_tokens"], Value::Null);
    assert_eq!(receipt["output_committed"], false);
    Ok(())
}

#[test]
fn completion_requires_matching_terminal_runtime_counts_and_nonempty_output() -> Result<()> {
    let mut observation = complete_observation();
    validate_completion(&observation, 16)?;
    observation.output.terminal = None;
    assert!(validate_completion(&observation, 16).is_err());
    observation = complete_observation();
    observation.timings.committed_generated_tokens = None;
    assert!(validate_completion(&observation, 16).is_err());
    observation = complete_observation();
    observation.timings.reused_tokens = Some(1);
    assert!(validate_completion(&observation, 16).is_err());
    observation = complete_observation();
    observation.output.text.clear();
    assert!(validate_completion(&observation, 16).is_err());
    Ok(())
}

#[test]
fn records_hash_exact_utf8_and_keep_unentered_counts_unknown() -> Result<()> {
    let selection = bounded_selection();
    let observation = complete_observation();
    let completed = record(&selection, &observation, true, Duration::from_secs(1))?;
    assert_eq!(completed["output_bytes"], "oak 🌳".len());
    assert_eq!(
        completed["output_sha256"],
        hex(hash_bytes("oak 🌳".as_bytes())?)
    );
    assert_eq!(completed["terminal_accounting_matches"], true);
    assert!(
        completed["first_text_delta_milliseconds"]
            .as_f64()
            .is_some_and(|value| (value - 1.0).abs() < f64::EPSILON)
    );
    let unentered = record(&selection, &Observation::default(), false, Duration::ZERO)?;
    for field in [
        "prompt_tokens",
        "generated_tokens",
        "reused_tokens",
        "output_text",
        "output_sha256",
        "runtime_milliseconds",
        "first_text_delta_milliseconds",
    ] {
        assert!(unentered[field].is_null());
    }
    assert_eq!(unentered["arm_succeeded"], false);
    assert_eq!(unentered["terminal_accounting_matches"], false);
    Ok(())
}

#[test]
fn failure_receipt_retains_provisional_output_without_committing_or_overwriting() -> Result<()> {
    let directory = std::env::temp_dir().join(format!(
        "cpu-bundle-receipt-{}-{}",
        std::process::id(),
        SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos(),
    ));
    fs::create_dir(&directory)?;
    let mut selection = bounded_selection();
    selection.evidence.clone_from(&directory);
    let started = Instant::now();
    let result = capture(&selection, started, |_, observation| {
        observation.stage = "failed-callback";
        observation.request_milliseconds = Some(1.0);
        observation.output.text = "partial".to_owned();
        Err(reject("injected output failure"))
    });
    assert!(result.is_err());
    let final_path = directory.join("cpu-bundle.json");
    let original = fs::read(&final_path)?;
    let record: Value = serde_json::from_slice(&original)?;
    assert_eq!(record["arm_succeeded"], false);
    assert_eq!(record["output_committed"], false);
    assert_eq!(record["output_text"], "partial");
    assert_eq!(record["terminal_delivered"], false);
    assert!(capture(&selection, started, |_, _| Err(reject("must not execute"))).is_err());
    assert_eq!(fs::read(&final_path)?, original);
    fs::remove_file(final_path)?;
    fs::remove_file(directory.join("cpu-bundle-started.json"))?;
    fs::remove_dir(directory)?;
    Ok(())
}
