#![forbid(unsafe_code)]
//! Focused serialized-artifact checks of the metadata-only tokenization driver.

#[cfg(feature = "cli")]
#[path = "support/tokenization_fixture.rs"]
mod tokenization_fixture;

#[cfg(feature = "cli")]
mod driver {
    use super::tokenization_fixture;
    use std::fs;
    use std::io::Write;
    use std::path::PathBuf;
    use std::process::{Command, Output, Stdio};
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    use bonsai_prism_runtime::Result;
    use bonsai_prism_runtime::model::Model;
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

        fn arguments(&self) -> Vec<String> {
            vec![
                "--model".into(),
                self.path.to_string_lossy().into_owned(),
                "--bytes".into(),
                self.bytes.len().to_string(),
                "--sha256".into(),
                self.hash(),
                "--memory-bytes".into(),
                HEADER_BUDGET.to_string(),
                "--context".into(),
                "4".into(),
                "--mode".into(),
                "tokenize".into(),
            ]
        }

        fn run(&self, arguments: &[String]) -> Result<Output> {
            Ok(Command::new(env!("CARGO_BIN_EXE_prism-inference"))
                .args(arguments)
                .current_dir(&self.root)
                .output()?)
        }

        fn record(&self, arguments: &[String]) -> Result<Value> {
            let result = self.run(arguments)?;
            assert!(
                result.status.success(),
                "{}",
                String::from_utf8_lossy(&result.stderr)
            );
            assert_eq!(result.stderr, [0_u8; 0]);
            assert_eq!(std::str::from_utf8(&result.stdout)?.lines().count(), 1);
            assert!(result.stdout.ends_with(b"\n"));
            Ok(serde_json::from_slice(&result.stdout)?)
        }

        fn run_stdin(&self, arguments: &[String], input: &[u8]) -> Result<Output> {
            let mut child = Command::new(env!("CARGO_BIN_EXE_prism-inference"))
                .args(arguments)
                .current_dir(&self.root)
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
    }

    impl Drop for Artifact {
        fn drop(&mut self) {
            // Only the exact file and directory created by this fixture are removed.
            let _ = fs::remove_file(&self.path);
            let _ = fs::remove_file(self.root.join("prompt-input.bin"));
            let _ = fs::remove_file(self.root.join("token-input.json"));
            let _ = fs::remove_dir(&self.root);
        }
    }

    fn replace(arguments: &mut [String], key: &str, value: &str) -> Result<()> {
        let index = arguments
            .iter()
            .position(|argument| argument == key)
            .ok_or("missing fixture option")?;
        value.clone_into(&mut arguments[index + 1]);
        Ok(())
    }

    fn rejected(fixture: &Artifact, arguments: &[String]) -> Result<String> {
        let result = fixture.run(arguments)?;
        assert!(!result.status.success());
        assert!(result.stdout.is_empty(), "failure emitted a partial record");
        Ok(String::from_utf8(result.stderr)?)
    }

    #[test]
    fn tokenize_driver_counts_text_without_materializing_weights_or_constructing_graph()
    -> Result<()> {
        let fixture = Artifact::new(260)?;
        assert!(fixture.bytes.len() > HEADER_BUDGET);
        let mut arguments = fixture.arguments();
        arguments.extend(["--prompt", "ab a!"].map(str::to_owned));
        let record = fixture.record(&arguments)?;
        assert_eq!(record["mode"], "tokenize");
        assert_eq!(record["input_mode"], "text");
        assert_eq!(record["token_ids"], json!([256, 32, 97, 33]));
        assert_eq!(record["token_count"], 4);
        assert_eq!(record["special_input"], "literal");
        assert_eq!(record["insert_special"], "none");
        assert_eq!(
            record["artifact"],
            json!({"bytes": fixture.bytes.len(), "sha256": fixture.hash()})
        );
        assert_eq!(record["context"], 4);
        assert_eq!(record["reserved_output_tokens"], 0);
        assert!(record.get("generated_tokens").is_none());
        let error = Model::load(
            &fixture.path,
            u64::try_from(fixture.bytes.len())?,
            &fixture.hash(),
            2 * 1024 * 1024,
        )
        .err()
        .ok_or("invalid executable payload was accepted")?;
        assert!(
            error
                .to_string()
                .contains("nonfinite executable tensor payload")
        );
        replace(&mut arguments, "--mode", "generate")?;
        arguments.extend(["--checkpoint-bytes", "4096", "--max-tokens", "1"].map(str::to_owned));
        assert!(
            rejected(&fixture, &arguments)?
                .contains("model tensor payloads exceed remaining memory budget")
        );
        Ok(())
    }

    #[test]
    fn tokenize_driver_keeps_raw_ids_and_special_input_policies_explicit() -> Result<()> {
        let fixture = Artifact::new(260)?;
        let mut arguments = fixture.arguments();
        arguments.extend(["--tokens", "0,258,259", "--max-tokens", "1"].map(str::to_owned));
        let record = fixture.record(&arguments)?;
        assert_eq!(record["input_mode"], "tokens");
        assert_eq!(record["token_ids"], json!([0, 258, 259]));
        assert_eq!(record["token_count"], 3);
        assert_eq!(record["special_input"], Value::Null);
        assert_eq!(record["reserved_output_tokens"], 1);
        replace(&mut arguments, "--tokens", "260")?;
        assert!(rejected(&fixture, &arguments)?.contains("outside the model vocabulary"));
        let mut arguments = fixture.arguments();
        replace(&mut arguments, "--context", "16")?;
        arguments.extend(["--prompt", "<stop>"].map(str::to_owned));
        let literal = fixture.record(&arguments)?;
        assert_eq!(literal["special_input"], "literal");
        assert_ne!(literal["token_ids"], json!([258]));
        arguments.extend(["--special-input", "parse"].map(str::to_owned));
        let parsed = fixture.record(&arguments)?;
        assert_eq!(parsed["special_input"], "parse");
        assert_eq!(parsed["token_ids"], json!([258]));
        assert_eq!(parsed["token_count"], 1);
        arguments.extend(["--insert-special", "model"].map(str::to_owned));
        let inserted = fixture.record(&arguments)?;
        assert_eq!(inserted["insert_special"], "model");
        assert_eq!(inserted["token_ids"], json!([259, 258, 258]));
        assert_eq!(inserted["token_count"], 3);
        replace(&mut arguments, "--context", "2")?;
        assert!(rejected(&fixture, &arguments)?.contains("token limit"));
        Ok(())
    }

    #[test]
    fn tokenize_driver_fails_atomically_for_context_identity_and_header_bounds() -> Result<()> {
        let fixture = Artifact::new(260)?;
        let mut arguments = fixture.arguments();
        arguments.extend(["--prompt", "ab a!", "--max-tokens", "1"].map(str::to_owned));
        assert!(rejected(&fixture, &arguments)?.contains("token limit"));
        let mut arguments = fixture.arguments();
        arguments.extend(["--prompt", "private prompt sentinel"].map(str::to_owned));
        for (key, value) in [
            ("--bytes", (fixture.bytes.len() + 1).to_string()),
            ("--sha256", "ab".repeat(32)),
            ("--memory-bytes", "1".to_owned()),
        ] {
            let mut invalid = arguments.clone();
            replace(&mut invalid, key, &value)?;
            let error = rejected(&fixture, &invalid)?;
            assert!(!error.contains("private prompt sentinel"));
        }
        let fixture = Artifact::new(261)?;
        let mut arguments = fixture.arguments();
        arguments.extend(["--prompt", "ab"].map(str::to_owned));
        assert!(rejected(&fixture, &arguments)?.contains("vocabulary"));
        Ok(())
    }

    #[test]
    fn tokenize_driver_prompt_file_and_stdin_match_exact_inline_text() -> Result<()> {
        let fixture = Artifact::new(260)?;
        let text = "\u{feff}ab\r\n\u{1f331}<stop>\n";
        let mut arguments = fixture.arguments();
        replace(&mut arguments, "--context", "64")?;
        arguments.extend(
            [
                "--prompt",
                text,
                "--special-input",
                "parse",
                "--insert-special",
                "model",
            ]
            .map(str::to_owned),
        );
        let inline = fixture.record(&arguments)?;
        let path = fixture.root.join("prompt-input.bin");
        fs::write(&path, text)?;
        let index = arguments
            .iter()
            .position(|key| key == "--prompt")
            .ok_or("missing prompt")?;
        arguments[index] = "--prompt-file".to_owned();
        arguments[index + 1] = path.to_string_lossy().into_owned();
        assert_eq!(fixture.record(&arguments)?, inline);
        arguments[index + 1] = "-".to_owned();
        let output = fixture.run_stdin(&arguments, text.as_bytes())?;
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(output.stderr, Vec::<u8>::new());
        assert_eq!(serde_json::from_slice::<Value>(&output.stdout)?, inline);
        Ok(())
    }

    #[test]
    fn tokenize_driver_prompt_files_reject_empty_invalid_and_oversized_input() -> Result<()> {
        let fixture = Artifact::new(260)?;
        let path = fixture.root.join("prompt-input.bin");
        let mut arguments = fixture.arguments();
        arguments.extend([
            "--prompt-file".to_owned(),
            path.to_string_lossy().into_owned(),
        ]);
        for bytes in [Vec::new(), vec![0xff], vec![b'a'; 1_048_577]] {
            fs::write(&path, bytes)?;
            assert!(!rejected(&fixture, &arguments)?.contains("prompt-input.bin"));
        }
        replace(&mut arguments, "--prompt-file", "relative.txt")?;
        assert!(rejected(&fixture, &arguments)?.contains("must be absolute"));
        replace(&mut arguments, "--prompt-file", "-")?;
        let output = fixture.run_stdin(&arguments, &[0xff])?;
        assert!(!output.status.success());
        assert_eq!(output.stdout, Vec::<u8>::new());
        Ok(())
    }

    #[test]
    fn tokenize_driver_token_file_and_stdin_match_exact_inline_ids() -> Result<()> {
        let fixture = Artifact::new(260)?;
        let mut arguments = fixture.arguments();
        replace(&mut arguments, "--context", "8")?;
        arguments.extend(["--tokens", "0,258,259,258", "--max-tokens", "4"].map(str::to_owned));
        let inline = fixture.record(&arguments)?;
        assert_eq!(inline["token_ids"], json!([0, 258, 259, 258]));
        assert_eq!(inline["token_count"], 4);
        assert_eq!(inline["insert_special"], "none");
        let serialized = b" \r\n[0, 258,259,258]\t\n";
        let path = fixture.root.join("token-input.json");
        fs::write(&path, serialized)?;
        let index = arguments
            .iter()
            .position(|key| key == "--tokens")
            .ok_or("missing tokens")?;
        arguments[index] = "--tokens-file".to_owned();
        arguments[index + 1] = path.to_string_lossy().into_owned();
        assert_eq!(fixture.record(&arguments)?, inline);
        arguments[index + 1] = "-".to_owned();
        let output = fixture.run_stdin(&arguments, serialized)?;
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(output.stderr, Vec::<u8>::new());
        assert_eq!(serde_json::from_slice::<Value>(&output.stdout)?, inline);
        let output = fixture.run_stdin(&arguments, b"[260]")?;
        assert!(!output.status.success());
        assert_eq!(output.stdout, Vec::<u8>::new());
        assert!(String::from_utf8(output.stderr)?.contains("outside the model vocabulary"));
        Ok(())
    }

    #[test]
    fn tokenize_driver_token_files_reject_privately_before_model_verification() -> Result<()> {
        let fixture = Artifact::new(260)?;
        let path = fixture.root.join("token-input.json");
        let mut arguments = fixture.arguments();
        replace(
            &mut arguments,
            "--model",
            &fixture.root.join("unopened.gguf").to_string_lossy(),
        )?;
        arguments.extend([
            "--tokens-file".to_owned(),
            path.to_string_lossy().into_owned(),
            "--max-tokens".to_owned(),
            "1".to_owned(),
        ]);
        for bytes in [
            Vec::new(),
            b"[]".to_vec(),
            b"[1,\"private token sentinel\"]".to_vec(),
            b"[4294967296]".to_vec(),
            b"[1] [2]".to_vec(),
            b"[0,258,259,0]".to_vec(),
            vec![b' '; 1_048_577],
        ] {
            fs::write(&path, &bytes)?;
            let error = rejected(&fixture, &arguments)?;
            assert!(!error.contains("private"));
            assert!(!error.contains("token-input.json"));
            assert!(!error.contains("unopened.gguf"));
            assert!(
                error.contains("raw token input")
                    || error.contains("raw input exceeds token limit")
            );
            replace(&mut arguments, "--tokens-file", "-")?;
            let output = fixture.run_stdin(&arguments, &bytes)?;
            assert!(!output.status.success());
            assert_eq!(output.stdout, Vec::<u8>::new());
            assert_eq!(String::from_utf8(output.stderr)?, error);
            replace(&mut arguments, "--tokens-file", &path.to_string_lossy())?;
        }
        replace(&mut arguments, "--tokens-file", "relative.json")?;
        assert!(rejected(&fixture, &arguments)?.contains("must be absolute"));
        fs::remove_file(&path)?;
        replace(&mut arguments, "--tokens-file", &path.to_string_lossy())?;
        let error = rejected(&fixture, &arguments)?;
        assert!(error.contains("failed to open raw token input"));
        assert!(!error.contains("token-input.json"));
        Ok(())
    }

    #[test]
    fn tokenize_driver_raw_ids_keep_metadata_only_contract_with_unsupported_text_tokenizer()
    -> Result<()> {
        let mut fixture = Artifact::new(260)?;
        let offset = fixture
            .bytes
            .windows(6)
            .position(|bytes| bytes == b"qwen35")
            .ok_or("missing pretokenizer fixture")?;
        fixture.bytes[offset..offset + 6].copy_from_slice(b"broken");
        fs::write(&fixture.path, &fixture.bytes)?;
        let mut arguments = fixture.arguments();
        arguments.extend(["--tokens", "0,258,259"].map(str::to_owned));
        let inline = fixture.record(&arguments)?;
        assert_eq!(inline["token_ids"], json!([0, 258, 259]));
        let index = arguments
            .iter()
            .position(|key| key == "--tokens")
            .ok_or("missing input")?;
        let path = fixture.root.join("token-input.json");
        let serialized = b"[0,258,259]";
        fs::write(&path, serialized)?;
        arguments[index] = "--tokens-file".to_owned();
        arguments[index + 1] = path.to_string_lossy().into_owned();
        assert_eq!(fixture.record(&arguments)?, inline);
        arguments[index + 1] = "-".to_owned();
        let output = fixture.run_stdin(&arguments, serialized)?;
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(output.stderr, Vec::<u8>::new());
        assert_eq!(serde_json::from_slice::<Value>(&output.stdout)?, inline);
        arguments[index] = "--prompt".to_owned();
        arguments[index + 1] = "ab".to_owned();
        assert!(rejected(&fixture, &arguments)?.contains("tokenizer.ggml.pre"));
        Ok(())
    }
}
