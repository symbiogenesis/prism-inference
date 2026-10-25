#![forbid(unsafe_code)]
//! Focused protocol and process checks of the persistent tokenization worker.

#[cfg(feature = "cli")]
#[path = "support/tokenization_fixture.rs"]
mod tokenization_fixture;

#[cfg(feature = "cli")]
mod worker {
    use super::tokenization_fixture;
    use std::fs;
    use std::io::{self, BufRead, BufReader, Cursor, Read, Write};
    use std::path::PathBuf;
    use std::process::{Command, Output, Stdio};
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    use bonsai_prism_runtime::Result;
    use bonsai_prism_runtime::special_input::SpecialInputPolicy;
    use bonsai_prism_runtime::text_output::Utf8Output;
    use bonsai_prism_runtime::text_session::TextOutputSession;
    use bonsai_prism_runtime::token_client::{ExpectedWorker, TokenClient};
    use bonsai_prism_runtime::token_worker::{MAX_WORKER_FRAME_BYTES, WorkerConfig, serve};
    use bonsai_prism_runtime::tokenization::{
        SpecialTokenInsertion, TextTokenizationPolicy, TokenBudget, TokenizationSession,
    };
    use serde_json::{Value, json};

    const HEADER_BUDGET: usize = 128 * 1024;
    static NEXT_ARTIFACT: AtomicU64 = AtomicU64::new(0);

    struct Artifact {
        root: PathBuf,
        path: PathBuf,
        bytes: Vec<u8>,
    }

    impl Artifact {
        fn new(vocab_size: u64) -> Result<Self> {
            let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
            let sequence = NEXT_ARTIFACT.fetch_add(1, Ordering::Relaxed);
            let root = std::env::temp_dir().join(format!(
                "SwiftBonsai-TokenizeDriver-{}-{nonce}-{sequence}",
                std::process::id()
            ));
            fs::create_dir(&root)?;
            let path = root.join("model.gguf");
            let bytes = tokenization_fixture::model_bytes(vocab_size, 258, "qwen35")?;
            fs::write(&path, &bytes)?;
            Ok(Self { root, path, bytes })
        }

        fn hash(&self) -> String {
            tokenization_fixture::sha256(&self.bytes)
        }
    }

    impl Drop for Artifact {
        fn drop(&mut self) {
            // Only the exact file and directory created by this fixture are removed.
            let _ = fs::remove_file(&self.path);
            let _ = fs::remove_file(self.root.join("prompt-input.bin"));
            let _ = fs::remove_dir(&self.root);
        }
    }

    const fn config() -> WorkerConfig {
        WorkerConfig {
            text_policy: TextTokenizationPolicy {
                budget: TokenBudget {
                    context: 4,
                    reserved_output_tokens: 1,
                },
                special_input: SpecialInputPolicy::Parse,
                insert_special: SpecialTokenInsertion::None,
            },
            max_frame_bytes: 1024,
        }
    }

    fn load(fixture: &Artifact) -> Result<TokenizationSession> {
        TokenizationSession::load(
            &fixture.path,
            u64::try_from(fixture.bytes.len())?,
            &fixture.hash(),
            HEADER_BUDGET,
        )
    }

    fn records(output: &[u8]) -> Result<Vec<Value>> {
        assert!(output.ends_with(b"\n"));
        std::str::from_utf8(output)?
            .lines()
            .map(|line| serde_json::from_str(line).map_err(Into::into))
            .collect()
    }

    fn run(
        session: &TokenizationSession,
        options: WorkerConfig,
        input: &[u8],
    ) -> Result<Vec<Value>> {
        let mut output = Vec::new();
        // One-byte reads exercise frame boundaries independently of chunk sizes.
        serve(
            session,
            options,
            &mut BufReader::with_capacity(1, input),
            &mut output,
        )?;
        records(&output)
    }

    #[test]
    fn persistent_counts_and_exact_ids_survive_artifact_removal() -> Result<()> {
        let fixture = Artifact::new(260)?;
        let session = load(&fixture)?;
        fs::remove_file(&fixture.path)?;
        let output = run(
            &session,
            config(),
            b"{\"id\":7,\"op\":\"count\",\"text\":\"ab\"}\n{\"id\":8,\"op\":\"encode\",\"text\":\"ab<stop>\"}\n{\"id\":9,\"op\":\"encode\",\"text\":\"\"}\n",
        )?;
        assert_eq!(
            output[0],
            json!({
                "type": "ready", "protocol": "bonsai-token-worker-v1",
                "model_sha256": fixture.hash(), "vocab_size": 260,
                "context": 4, "reserved_output_tokens": 1,
                "max_frame_bytes": 1024, "special_input": "parse", "insert_special": "none"
            })
        );
        assert_eq!(
            output[1],
            json!({"type": "count", "id": 7, "prompt_token_count": 1})
        );
        assert_eq!(
            output[2],
            json!({"type": "encoded", "id": 8, "prompt_token_count": 2, "token_ids": [256, 258]})
        );
        assert_eq!(
            output[3],
            json!({"type": "encoded", "id": 9, "prompt_token_count": 0, "token_ids": []})
        );
        assert_eq!(output.len(), 4);
        Ok(())
    }

    #[test]
    fn verified_service_boundaries_compose_after_artifact_release() -> Result<()> {
        let fixture = Artifact::new(260)?;
        let tokenizer = load(&fixture)?;
        let output = TextOutputSession::load(
            &fixture.path,
            u64::try_from(fixture.bytes.len())?,
            &fixture.hash(),
            HEADER_BUDGET,
        )?;
        fs::remove_file(&fixture.path)?;
        let requests = b"{\"id\":0,\"op\":\"count\",\"text\":\"ab\"}\n{\"id\":1,\"op\":\"encode\",\"text\":\"ab<stop>\"}\n";
        let mut responses = Vec::new();
        serve(
            &tokenizer,
            config(),
            &mut Cursor::new(requests),
            &mut responses,
        )?;
        let hash = fixture.hash();
        let mut sent = Vec::new();
        let mut client = TokenClient::connect(
            BufReader::with_capacity(1, responses.as_slice()),
            &mut sent,
            ExpectedWorker {
                model_sha256: &hash,
                vocab_size: 260,
                config: config(),
                max_response_frame_bytes: 1024,
            },
        )?;
        assert_eq!(client.count("ab")?, 1);
        let encoded = client.encode("ab<stop>")?;
        assert_eq!(encoded.token_ids, [256, 258]);
        assert_eq!(encoded.context, 4);
        assert_eq!(encoded.reserved_output_tokens, 1);
        assert_eq!(output.identity(), hash);
        let mut lossless = Vec::new();
        let mut display = Utf8Output::new();
        let mut text = String::new();
        for id in encoded.token_ids {
            let piece = output.piece(id)?;
            lossless.extend_from_slice(piece);
            text.push_str(&display.push(piece)?);
        }
        text.push_str(display.finish());
        assert_eq!(lossless, b"ab<stop>");
        assert_eq!(text, "ab<stop>");
        let _ = client.into_parts();
        assert_eq!(sent, requests);
        Ok(())
    }

    #[test]
    fn raw_ids_preserve_exact_order_without_text_policy_or_insertion() -> Result<()> {
        let fixture = Artifact::new(260)?;
        let session = load(&fixture)?;
        fs::remove_file(&fixture.path)?;
        let mut options = config();
        options.text_policy.insert_special = SpecialTokenInsertion::Model;
        let output = run(
            &session,
            options,
            b"{\"id\":1,\"op\":\"encode_tokens\",\"token_ids\":[258,0,259]}\n{\"id\":2,\"op\":\"count_tokens\",\"token_ids\":[258,0,259]}\n{\"id\":3,\"op\":\"encode_tokens\",\"token_ids\":[]}\n{\"id\":4,\"op\":\"encode\",\"text\":\"ab\"}\n",
        )?;
        assert_eq!(output.len(), 5);
        assert_eq!(
            output[1],
            json!({"type": "encoded", "id": 1, "prompt_token_count": 3, "token_ids": [258, 0, 259]})
        );
        assert_eq!(
            output[2],
            json!({"type": "count", "id": 2, "prompt_token_count": 3})
        );
        assert_eq!(
            output[3],
            json!({"type": "encoded", "id": 3, "prompt_token_count": 0, "token_ids": []})
        );
        assert_eq!(output[4]["token_ids"], json!([259, 256, 258]));
        Ok(())
    }

    #[test]
    fn raw_ids_reject_vocabulary_and_context_overflow_then_recover() -> Result<()> {
        let fixture = Artifact::new(260)?;
        let output = run(
            &load(&fixture)?,
            config(),
            b"{\"id\":1,\"op\":\"count_tokens\",\"token_ids\":[260]}\n{\"id\":2,\"op\":\"encode_tokens\",\"token_ids\":[0,1,2,3]}\n{\"id\":3,\"op\":\"encode_tokens\",\"token_ids\":[4294967295]}\n{\"id\":4,\"op\":\"encode_tokens\",\"token_ids\":[259,259,0]}\n",
        )?;
        for (id, record) in output.iter().enumerate().take(4).skip(1) {
            assert_eq!(
                record,
                &json!({"type": "error", "id": id, "code": "rejected_input"})
            );
        }
        assert_eq!(output[4]["token_ids"], json!([259, 259, 0]));
        Ok(())
    }

    #[test]
    fn raw_ids_require_exclusive_strict_operation_shapes() -> Result<()> {
        let fixture = Artifact::new(260)?;
        let invalid = [
            r#"{"id":1,"op":"count_tokens","text":"private-prompt"}"#,
            r#"{"id":1,"op":"encode_tokens","token_ids":[],"text":"private-prompt"}"#,
            r#"{"id":1,"op":"count","token_ids":[]}"#,
            r#"{"id":1,"op":"count","text":"private-prompt","token_ids":[]}"#,
            r#"{"id":1,"op":"encode_tokens","token_ids":[-1]}"#,
            r#"{"id":1,"op":"encode_tokens","token_ids":[1.5]}"#,
            r#"{"id":1,"op":"encode_tokens","token_ids":[4294967296]}"#,
            r#"{"id":1,"op":"encode_tokens","token_ids":["1"]}"#,
            r#"{"id":1,"op":"encode_tokens","token_ids":null}"#,
            r#"{"id":1,"op":"encode_tokens","token_ids":[],"token_ids":[1]}"#,
            r#"{"id":1,"id":2,"op":"count_tokens","token_ids":[]}"#,
            r#"{"id":1,"op":"count_tokens","op":"encode_tokens","token_ids":[]}"#,
            r#"{"id":1,"op":"count_tokens","token_ids":[],"context":99}"#,
        ];
        let mut input = invalid.join("\n");
        input.push_str("\n{\"id\":2,\"op\":\"count_tokens\",\"token_ids\":[0]}\n");
        let output = run(&load(&fixture)?, config(), input.as_bytes())?;
        assert_eq!(output.len(), invalid.len() + 2);
        for record in &output[1..=invalid.len()] {
            assert_eq!(
                record,
                &json!({"type": "error", "id": null, "code": "invalid_request"})
            );
        }
        assert_eq!(
            output[invalid.len() + 1],
            json!({"type": "count", "id": 2, "prompt_token_count": 1})
        );
        Ok(())
    }

    #[test]
    fn fixed_policy_and_reservation_reject_overflow_then_recover() -> Result<()> {
        let fixture = Artifact::new(260)?;
        let session = load(&fixture)?;
        let mut options = config();
        options.text_policy.insert_special = SpecialTokenInsertion::Model;
        let output = run(
            &session,
            options,
            b"{\"id\":0,\"op\":\"encode\",\"text\":\"ab<stop>\"}\n{\"id\":1,\"op\":\"encode\",\"text\":\"ab\"}\n{\"id\":2,\"op\":\"encode\",\"text\":\"\"}\n",
        )?;
        assert_eq!(
            output[1],
            json!({"type": "error", "id": 0, "code": "rejected_input"})
        );
        assert_eq!(
            output[2],
            json!({"type": "encoded", "id": 1, "prompt_token_count": 3, "token_ids": [259, 256, 258]})
        );
        assert_eq!(output[3]["token_ids"], json!([259, 258]));
        options.text_policy.special_input = SpecialInputPolicy::Literal;
        options.text_policy.insert_special = SpecialTokenInsertion::None;
        options.text_policy.budget.context = 16;
        let output = run(
            &session,
            options,
            b"{\"id\":0,\"op\":\"encode\",\"text\":\"<stop>\"}\n",
        )?;
        assert_eq!(output[0]["special_input"], "literal");
        assert_eq!(output[1]["token_ids"], json!([60, 115, 116, 111, 112, 62]));
        Ok(())
    }

    #[test]
    fn malformed_requests_are_private_and_do_not_poison_later_frames() -> Result<()> {
        let fixture = Artifact::new(260)?;
        let session = load(&fixture)?;
        let invalid: &[&[u8]] = &[
            b"",
            b"\xff",
            b"null",
            b"[]",
            br#"[1,"count","ab"]"#,
            b"{",
            b"{}",
            br#"{"id":1,"op":"count","text":"private-prompt","text":"a"}"#,
            br#"{"id":1,"id":2,"op":"count","text":"private-prompt"}"#,
            br#"{"id":1,"op":"count","op":"encode","text":"private-prompt"}"#,
            br#"{"id":1,"op":"count","text":"private-prompt","context":99}"#,
            br#"{"id":-1,"op":"count","text":"private-prompt"}"#,
            br#"{"id":1.5,"op":"count","text":"private-prompt"}"#,
            br#"{"id":"1","op":"count","text":"private-prompt"}"#,
            br#"{"id":18446744073709551616,"op":"count","text":"private-prompt"}"#,
            br#"{"id":1,"op":"count","text":null}"#,
            br#"{"id":1,"op":"count"}"#,
            br#"{"id":1,"op":"generate","text":"private-prompt"}"#,
            br#"{"id":1,"op":"count","text":"private-prompt"} {}"#,
        ];
        let mut input = Vec::new();
        for frame in invalid {
            input.extend_from_slice(frame);
            input.push(b'\n');
        }
        input.extend_from_slice(
            b"{\"id\":18446744073709551615,\"op\":\"count\",\"text\":\"ab\"}\r\n",
        );
        let mut output = Vec::new();
        serve(&session, config(), &mut Cursor::new(&input), &mut output)?;
        assert!(!std::str::from_utf8(&output)?.contains("private-prompt"));
        let output = records(&output)?;
        assert_eq!(output.len(), invalid.len() + 2);
        for record in &output[1..=invalid.len()] {
            assert_eq!(
                record,
                &json!({"type": "error", "id": null, "code": "invalid_request"})
            );
        }
        assert_eq!(
            output[invalid.len() + 1],
            json!({"type": "count", "id": u64::MAX, "prompt_token_count": 1})
        );
        Ok(())
    }

    #[test]
    fn newline_bound_and_unterminated_eof_have_explicit_results() -> Result<()> {
        let fixture = Artifact::new(260)?;
        let session = load(&fixture)?;
        let frame = b"{\"id\":1,\"op\":\"count\",\"text\":\"ab\"}";
        let mut options = config();
        options.max_frame_bytes = frame.len();
        let mut input = frame.to_vec();
        input.push(b'\n');
        assert_eq!(run(&session, options, &input)?[1]["type"], "count");
        input.extend_from_slice(frame);
        let output = run(&session, options, &input)?;
        assert_eq!(output.len(), 3);
        assert_eq!(
            output[2],
            json!({"type": "error", "id": null, "code": "invalid_request"})
        );
        assert_eq!(run(&session, options, b"")?.len(), 1);
        // One extra byte is fatal, even if followed by a valid request.
        let mut oversized = frame.to_vec();
        oversized.extend_from_slice(b" \n");
        oversized.extend_from_slice(&input);
        let mut output = Vec::new();
        assert!(
            serve(
                &session,
                options,
                &mut BufReader::with_capacity(1, oversized.as_slice()),
                &mut output
            )
            .is_err()
        );
        assert_eq!(records(&output)?.len(), 1);
        Ok(())
    }

    #[test]
    fn invalid_configuration_never_emits_readiness() -> Result<()> {
        let fixture = Artifact::new(260)?;
        let session = load(&fixture)?;
        let mut options = config();
        for bytes in [0, MAX_WORKER_FRAME_BYTES + 1] {
            options.max_frame_bytes = bytes;
            let mut output = Vec::new();
            assert!(serve(&session, options, &mut Cursor::new([]), &mut output).is_err());
            assert_eq!(output, [0_u8; 0]);
        }
        options = config();
        options.text_policy.budget.reserved_output_tokens = options.text_policy.budget.context;
        let mut output = Vec::new();
        assert!(serve(&session, options, &mut Cursor::new([]), &mut output).is_err());
        assert_eq!(output, [0_u8; 0]);
        Ok(())
    }

    struct BrokenRead;

    impl Read for BrokenRead {
        fn read(&mut self, _buffer: &mut [u8]) -> io::Result<usize> {
            Err(io::Error::other("private transport detail"))
        }
    }

    struct BrokenFlush;

    impl Write for BrokenFlush {
        fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
            Ok(buffer.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Err(io::Error::other("private transport detail"))
        }
    }

    #[test]
    fn transport_read_and_flush_failures_terminate() -> Result<()> {
        let fixture = Artifact::new(260)?;
        let session = load(&fixture)?;
        let mut output = Vec::new();
        assert!(
            serve(
                &session,
                config(),
                &mut BufReader::new(BrokenRead),
                &mut output
            )
            .is_err()
        );
        assert_eq!(records(&output)?.len(), 1);
        assert!(serve(&session, config(), &mut Cursor::new([]), &mut BrokenFlush).is_err());
        Ok(())
    }

    fn arguments(fixture: &Artifact) -> Vec<String> {
        vec![
            "--model".into(),
            fixture.path.to_string_lossy().into_owned(),
            "--bytes".into(),
            fixture.bytes.len().to_string(),
            "--sha256".into(),
            fixture.hash(),
            "--memory-bytes".into(),
            HEADER_BUDGET.to_string(),
            "--context".into(),
            "4".into(),
            "--reserved-output-tokens".into(),
            "1".into(),
            "--special-input".into(),
            "parse".into(),
            "--insert-special".into(),
            "none".into(),
            "--max-frame-bytes".into(),
            "1024".into(),
        ]
    }

    fn child(fixture: &Artifact, options: &[String], input: &[u8]) -> Result<Output> {
        let mut child = Command::new(env!("CARGO_BIN_EXE_bonsai-token-worker"))
            .args(options)
            .current_dir(&fixture.root)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;
        child
            .stdin
            .take()
            .ok_or("missing child stdin")?
            .write_all(input)?;
        Ok(child.wait_with_output()?)
    }

    #[test]
    fn raw_ids_binary_uses_the_same_verified_budget_and_response_schema() -> Result<()> {
        let fixture = Artifact::new(260)?;
        let output = child(
            &fixture,
            &arguments(&fixture),
            b"{\"id\":0,\"op\":\"count_tokens\",\"token_ids\":[0,1,2,3]}\n{\"id\":1,\"op\":\"encode_tokens\",\"token_ids\":[259,0,258]}\n",
        )?;
        assert!(output.status.success());
        assert_eq!(output.stderr, [0_u8; 0]);
        let output = records(&output.stdout)?;
        assert_eq!(output.len(), 3);
        assert_eq!(output[0]["model_sha256"], fixture.hash());
        assert_eq!(
            output[1],
            json!({"type": "error", "id": 0, "code": "rejected_input"})
        );
        assert_eq!(
            output[2],
            json!({"type": "encoded", "id": 1, "prompt_token_count": 3, "token_ids": [259, 0, 258]})
        );
        Ok(())
    }

    #[test]
    fn binary_flushes_readiness_before_input_then_reuses_verified_session() -> Result<()> {
        let fixture = Artifact::new(260)?;
        let mut child = Command::new(env!("CARGO_BIN_EXE_bonsai-token-worker"))
            .args(arguments(&fixture))
            .current_dir(&fixture.root)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;
        let mut stdout = BufReader::new(child.stdout.take().ok_or("missing child stdout")?);
        let mut ready = String::new();
        stdout.read_line(&mut ready)?;
        assert_eq!(
            serde_json::from_str::<Value>(&ready)?["model_sha256"],
            fixture.hash()
        );
        fs::remove_file(&fixture.path)?;
        child.stdin.take().ok_or("missing child stdin")?.write_all(b"{\"id\":1,\"op\":\"count\",\"text\":\"ab\"}\n{\"id\":2,\"op\":\"encode\",\"text\":\"ab\"}\n")?;
        let mut responses = Vec::new();
        stdout.read_to_end(&mut responses)?;
        let result = child.wait_with_output()?;
        assert!(result.status.success());
        assert_eq!(result.stderr, [0_u8; 0]);
        let responses = records(&responses)?;
        assert_eq!(responses.len(), 2);
        assert_eq!(
            responses[0],
            json!({"type": "count", "id": 1, "prompt_token_count": 1})
        );
        assert_eq!(responses[1]["token_ids"], json!([256]));
        Ok(())
    }

    #[test]
    fn binary_rejects_unsafe_startup_and_fatal_frames_with_static_diagnostics() -> Result<()> {
        let fixture = Artifact::new(260)?;
        let base = arguments(&fixture);
        for tail in [
            vec!["--context", "2"],
            vec!["--prompt", "private-prompt"],
            vec!["--unknown", "private-prompt"],
            vec!["--max-frame-bytes"],
            vec!["--help"],
        ] {
            let mut options = base.clone();
            options.extend(tail.into_iter().map(str::to_owned));
            let output = child(&fixture, &options, b"")?;
            assert!(!output.status.success());
            assert_eq!(output.stdout, [0_u8; 0]);
            assert_eq!(output.stderr, b"token worker startup failed\n");
        }
        let mut options = base.clone();
        let digest = options
            .iter()
            .position(|item| item == "--sha256")
            .ok_or("missing hash")?;
        options[digest + 1] = "0".repeat(64);
        let output = child(&fixture, &options, b"")?;
        assert!(!output.status.success());
        assert_eq!(output.stdout, [0_u8; 0]);
        assert_eq!(output.stderr, b"token worker startup failed\n");
        for (key, value) in [
            ("--model", "relative.gguf"),
            ("--bytes", "0"),
            ("--bytes", "+1"),
            ("--memory-bytes", "0"),
            ("--context", "0"),
            ("--context", "262145"),
            ("--reserved-output-tokens", "4"),
            ("--reserved-output-tokens", "-1"),
            ("--special-input", "auto"),
            ("--insert-special", "auto"),
            ("--max-frame-bytes", "0"),
            ("--max-frame-bytes", "1048577"),
        ] {
            let mut options = base.clone();
            let position = options
                .iter()
                .position(|option| option == key)
                .ok_or("missing option")?;
            options[position + 1] = value.to_owned();
            let output = child(&fixture, &options, b"")?;
            assert!(!output.status.success());
            assert_eq!(output.stdout, [0_u8; 0]);
            assert_eq!(output.stderr, b"token worker startup failed\n");
        }
        let help = child(&fixture, &["--help".into()], b"")?;
        assert!(help.status.success());
        assert_eq!(help.stderr, [0_u8; 0]);
        assert!(std::str::from_utf8(&help.stdout)?.starts_with("Usage: bonsai-token-worker"));
        let output = child(&fixture, &base, &vec![b'x'; 1025])?;
        assert!(!output.status.success());
        assert_eq!(records(&output.stdout)?.len(), 1);
        assert_eq!(output.stderr, b"token worker transport failed\n");
        Ok(())
    }
}
