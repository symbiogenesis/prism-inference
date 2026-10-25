#![forbid(unsafe_code)]
//! Focused readiness, framing and fail-closed client protocol checks.

#[cfg(feature = "cli")]
mod client {
    use std::io::{self, BufReader, Cursor, Read, Write};

    use bonsai_prism_runtime::Result;
    use bonsai_prism_runtime::special_input::SpecialInputPolicy;
    use bonsai_prism_runtime::token_client::{ClientError, ExpectedWorker, TokenClient};
    use bonsai_prism_runtime::token_worker::{MAX_WORKER_FRAME_BYTES, WorkerConfig};
    use bonsai_prism_runtime::tokenization::{
        SpecialTokenInsertion, TextTokenizationPolicy, TokenBudget,
    };
    use serde_json::{Value, json};

    const HASH: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    const fn config() -> WorkerConfig {
        WorkerConfig {
            text_policy: TextTokenizationPolicy {
                budget: TokenBudget {
                    context: 4,
                    reserved_output_tokens: 1,
                },
                special_input: SpecialInputPolicy::Literal,
                insert_special: SpecialTokenInsertion::None,
            },
            max_frame_bytes: 1024,
        }
    }

    const fn expected(options: WorkerConfig) -> ExpectedWorker<'static> {
        ExpectedWorker {
            model_sha256: HASH,
            vocab_size: 260,
            config: options,
            max_response_frame_bytes: 1024,
        }
    }

    fn ready(options: WorkerConfig) -> Value {
        json!({
            "type": "ready", "protocol": "bonsai-token-worker-v1",
            "model_sha256": HASH, "vocab_size": 260,
            "context": options.text_policy.budget.context,
            "reserved_output_tokens": options.text_policy.budget.reserved_output_tokens,
            "max_frame_bytes": options.max_frame_bytes,
            "special_input": match options.text_policy.special_input {
                SpecialInputPolicy::Literal => "literal",
                SpecialInputPolicy::Parse => "parse",
            },
            "insert_special": match options.text_policy.insert_special {
                SpecialTokenInsertion::None => "none",
                SpecialTokenInsertion::Model => "model",
            },
        })
    }

    fn script(options: WorkerConfig, responses: &[&[u8]]) -> Result<Vec<u8>> {
        let mut bytes = serde_json::to_vec(&ready(options))?;
        bytes.push(b'\n');
        for response in responses {
            bytes.extend_from_slice(response);
        }
        Ok(bytes)
    }

    #[test]
    fn checks_exact_text_counts_ids_and_recoverable_rejection() -> Result<()> {
        let input = script(
            config(),
            &[
                b"{\"type\":\"count\",\"id\":0,\"prompt_token_count\":1}\n",
                b"{\"type\":\"error\",\"id\":1,\"code\":\"rejected_input\"}\n",
                b"{\"type\":\"encoded\",\"id\":2,\"prompt_token_count\":3,\"token_ids\":[259,256,258]}\n",
                b"{\"type\":\"encoded\",\"id\":3,\"prompt_token_count\":0,\"token_ids\":[]}\n",
            ],
        )?;
        let mut output = Vec::new();
        let mut client = TokenClient::connect(
            BufReader::with_capacity(1, input.as_slice()),
            &mut output,
            expected(config()),
        )?;
        assert!(client.is_usable());
        let private = "private \"text\"\n\r\t🦀";
        assert_eq!(client.count(private)?, 1);
        assert_eq!(client.encode(private), Err(ClientError::RejectedInput));
        assert!(client.is_usable());
        let encoded = client.encode("ab")?;
        assert_eq!(encoded.token_ids, [259, 256, 258]);
        assert_eq!(encoded.context, 4);
        assert_eq!(encoded.reserved_output_tokens, 1);
        assert_eq!(client.encode("")?.prompt_token_count(), 0);
        let _ = client.into_parts();
        let requests = std::str::from_utf8(&output)?
            .lines()
            .map(serde_json::from_str::<Value>)
            .collect::<std::result::Result<Vec<_>, _>>()?;
        assert_eq!(
            requests,
            [
                json!({"id": 0, "op": "count", "text": private}),
                json!({"id": 1, "op": "encode", "text": private}),
                json!({"id": 2, "op": "encode", "text": "ab"}),
                json!({"id": 3, "op": "encode", "text": ""}),
            ]
        );
        Ok(())
    }

    #[test]
    fn raw_ids_preserve_empty_duplicates_and_accounting_with_text_policy_pinned() -> Result<()> {
        let mut options = config();
        options.text_policy.special_input = SpecialInputPolicy::Parse;
        options.text_policy.insert_special = SpecialTokenInsertion::Model;
        let input = script(
            options,
            &[
                b"{\"type\":\"count\",\"id\":0,\"prompt_token_count\":0}\n",
                b"{\"type\":\"encoded\",\"id\":1,\"prompt_token_count\":3,\"token_ids\":[259,0,259]}\n",
                b"{\"type\":\"error\",\"id\":2,\"code\":\"rejected_input\"}\n",
                b"{\"type\":\"count\",\"id\":3,\"prompt_token_count\":1}\n",
                b"{\"type\":\"error\",\"id\":4,\"code\":\"rejected_input\"}\n",
                b"{\"type\":\"encoded\",\"id\":5,\"prompt_token_count\":0,\"token_ids\":[]}\n",
            ],
        )?;
        let mut output = Vec::new();
        let mut client = TokenClient::connect(
            BufReader::with_capacity(1, input.as_slice()),
            &mut output,
            expected(options),
        )?;
        assert_eq!(client.count_tokens(&[])?, 0);
        let encoded = client.encode_tokens(&[259, 0, 259])?;
        assert_eq!(encoded.token_ids, [259, 0, 259]);
        assert_eq!(encoded.context, 4);
        assert_eq!(encoded.reserved_output_tokens, 1);
        assert_eq!(client.count_tokens(&[259]), Err(ClientError::RejectedInput));
        assert!(client.is_usable());
        assert_eq!(client.count("a")?, 1);
        assert_eq!(client.encode_tokens(&[0]), Err(ClientError::RejectedInput));
        assert!(client.is_usable());
        assert_eq!(client.encode_tokens(&[])?.token_ids, [0_u32; 0]);
        let _ = client.into_parts();
        let requests = std::str::from_utf8(&output)?
            .lines()
            .map(serde_json::from_str::<Value>)
            .collect::<std::result::Result<Vec<_>, _>>()?;
        assert_eq!(
            requests,
            [
                json!({"id": 0, "op": "count_tokens", "token_ids": []}),
                json!({"id": 1, "op": "encode_tokens", "token_ids": [259, 0, 259]}),
                json!({"id": 2, "op": "count_tokens", "token_ids": [259]}),
                json!({"id": 3, "op": "count", "text": "a"}),
                json!({"id": 4, "op": "encode_tokens", "token_ids": [0]}),
                json!({"id": 5, "op": "encode_tokens", "token_ids": []}),
            ]
        );
        Ok(())
    }

    #[test]
    fn raw_local_vocabulary_context_and_frame_bounds_leave_transport_and_id_reusable() -> Result<()>
    {
        for (encode, request, response) in [
            (
                false,
                b"{\"id\":0,\"op\":\"count_tokens\",\"token_ids\":[259]}".as_slice(),
                b"{\"type\":\"count\",\"id\":0,\"prompt_token_count\":1}\n".as_slice(),
            ),
            (
                true,
                b"{\"id\":0,\"op\":\"encode_tokens\",\"token_ids\":[259]}",
                b"{\"type\":\"encoded\",\"id\":0,\"prompt_token_count\":1,\"token_ids\":[259]}\n",
            ),
        ] {
            let mut options = config();
            options.max_frame_bytes = request.len();
            let input = script(options, &[response])?;
            let mut output = Vec::new();
            let mut client =
                TokenClient::connect(Cursor::new(input), &mut output, expected(options))?;
            for tokens in [&[260][..], &[u32::MAX][..], &[0, 0, 0, 0][..]] {
                let error = if encode {
                    client.encode_tokens(tokens).err()
                } else {
                    client.count_tokens(tokens).err()
                };
                assert_eq!(error, Some(ClientError::RejectedInput));
                assert!(client.is_usable());
            }
            let error = if encode {
                client.encode_tokens(&[259, 259]).err()
            } else {
                client.count_tokens(&[259, 259]).err()
            };
            assert_eq!(error, Some(ClientError::RequestTooLarge));
            assert!(client.is_usable());
            if encode {
                assert_eq!(client.encode_tokens(&[259])?.token_ids, [259]);
            } else {
                assert_eq!(client.count_tokens(&[259])?, 1);
            }
            let _ = client.into_parts();
            let mut expected_request = request.to_vec();
            expected_request.push(b'\n');
            assert_eq!(output, expected_request);
        }
        Ok(())
    }

    #[test]
    fn raw_responses_require_exact_count_sequence_shape_and_correlation() -> Result<()> {
        for (encode, response) in [
            (false, b"{\"type\":\"count\",\"id\":0,\"prompt_token_count\":2}\n".as_slice()),
            (false, b"{\"type\":\"count\",\"id\":1,\"prompt_token_count\":3}\n"),
            (false, b"{\"type\":\"encoded\",\"id\":0,\"prompt_token_count\":3,\"token_ids\":[259,0,259]}\n"),
            (true, b"{\"type\":\"encoded\",\"id\":0,\"prompt_token_count\":3,\"token_ids\":[259,259,0]}\n"),
            (true, b"{\"type\":\"encoded\",\"id\":0,\"prompt_token_count\":3,\"token_ids\":[259,0,258]}\n"),
            (true, b"{\"type\":\"encoded\",\"id\":0,\"prompt_token_count\":2,\"token_ids\":[259,0]}\n"),
            (true, b"{\"type\":\"encoded\",\"id\":0,\"prompt_token_count\":3,\"token_ids\":[259,0,260]}\n"),
            (true, b"{\"type\":\"encoded\",\"id\":1,\"prompt_token_count\":3,\"token_ids\":[259,0,259]}\n"),
            (true, b"{\"type\":\"count\",\"id\":0,\"prompt_token_count\":3}\n"),
            (true, b"{\"type\":\"error\",\"id\":1,\"code\":\"rejected_input\"}\n"),
        ] {
            let input = script(config(), &[response])?;
            let mut output = Vec::new();
            let mut client = TokenClient::connect(Cursor::new(input), &mut output, expected(config()))?;
            let error = if encode {
                client.encode_tokens(&[259, 0, 259]).err()
            } else {
                client.count_tokens(&[259, 0, 259]).err()
            };
            assert_eq!(error, Some(ClientError::InvalidResponse));
            assert!(!client.is_usable());
            assert_eq!(client.count_tokens(&[]), Err(ClientError::Unavailable));
            assert_eq!(client.encode_tokens(&[260]), Err(ClientError::Unavailable));
            assert_eq!(client.count("a"), Err(ClientError::Unavailable));
            let _ = client.into_parts();
            assert_eq!(std::str::from_utf8(&output)?.lines().count(), 1);
        }
        Ok(())
    }

    #[test]
    fn readiness_requires_every_pinned_identity_and_policy_field() -> Result<()> {
        for (field, bad) in [
            ("protocol", json!("bonsai-token-worker-v2")),
            ("model_sha256", json!("0".repeat(64))),
            ("vocab_size", json!(259)),
            ("context", json!(5)),
            ("reserved_output_tokens", json!(0)),
            ("max_frame_bytes", json!(1023)),
            ("special_input", json!("parse")),
            ("insert_special", json!("model")),
        ] {
            let mut value = ready(config());
            value[field] = bad;
            let mut bytes = serde_json::to_vec(&value)?;
            bytes.push(b'\n');
            let mut output = Vec::new();
            assert!(matches!(
                TokenClient::connect(Cursor::new(bytes), &mut output, expected(config())),
                Err(ClientError::ReadinessMismatch)
            ));
            assert_eq!(output, [0_u8; 0]);
        }
        // Both alternate explicit policies are also accepted when pinned.
        let mut options = config();
        options.text_policy.special_input = SpecialInputPolicy::Parse;
        options.text_policy.insert_special = SpecialTokenInsertion::Model;
        let bytes = script(options, &[])?;
        let client = TokenClient::connect(Cursor::new(bytes), Vec::new(), expected(options))?;
        assert!(client.is_usable());
        Ok(())
    }

    #[test]
    fn readiness_rejects_unknown_duplicate_missing_fields_and_bad_frames() -> Result<()> {
        let original = serde_json::to_string(&ready(config()))?;
        let unknown = original.replacen('{', "{\"private-unknown\":true,", 1);
        let duplicate = original.replacen('{', "{\"vocab_size\":260,", 1);
        let duplicate_tag = original.replacen('{', "{\"type\":\"ready\",", 1);
        let mut missing = ready(config());
        missing
            .as_object_mut()
            .ok_or("missing ready object")?
            .remove("context");
        let cases = [
            format!("{unknown}\n").into_bytes(),
            format!("{duplicate}\n").into_bytes(),
            format!("{duplicate_tag}\n").into_bytes(),
            format!("{missing}\n").into_bytes(),
            b"{\"type\":\"count\",\"id\":0,\"prompt_token_count\":1}\n".to_vec(),
            b"[]\n".to_vec(),
            b"\n".to_vec(),
            vec![0xff, b'\n'],
            original.as_bytes().to_vec(),
            Vec::new(),
        ];
        for bytes in cases {
            let mut output = Vec::new();
            assert!(
                TokenClient::connect(Cursor::new(bytes), &mut output, expected(config())).is_err()
            );
            assert_eq!(output, [0_u8; 0]);
        }
        Ok(())
    }

    #[test]
    fn invalid_local_configuration_does_not_touch_transport() {
        let mut cases = Vec::new();
        let uppercase = "A".repeat(64);
        let nonhex = "g".repeat(64);
        for hash in ["", "private", &uppercase, &nonhex] {
            let mut value = expected(config());
            value.model_sha256 = hash;
            cases.push(value);
        }
        for vocab_size in [0, 1_000_001] {
            let mut value = expected(config());
            value.vocab_size = vocab_size;
            cases.push(value);
        }
        for limit in [0, MAX_WORKER_FRAME_BYTES + 1] {
            let mut value = expected(config());
            value.max_response_frame_bytes = limit;
            cases.push(value);
            value = expected(config());
            value.config.max_frame_bytes = limit;
            cases.push(value);
        }
        for context in [0, 262_145] {
            let mut value = expected(config());
            value.config.text_policy.budget.context = context;
            cases.push(value);
        }
        let mut value = expected(config());
        value.config.text_policy.budget.reserved_output_tokens = 4;
        cases.push(value);
        for value in cases {
            let mut reader = FailingRead { calls: 0 };
            let mut output = Vec::new();
            assert!(matches!(
                TokenClient::connect(BufReader::new(&mut reader), &mut output, value),
                Err(ClientError::InvalidConfiguration)
            ));
            assert_eq!(reader.calls, 0);
            assert_eq!(output, [0_u8; 0]);
        }
    }

    #[test]
    fn serialized_request_bound_is_exact_and_rejection_does_not_consume_an_id() -> Result<()> {
        let mut options = config();
        let request = b"{\"id\":0,\"op\":\"count\",\"text\":\"ab\"}";
        options.max_frame_bytes = request.len();
        let input = script(
            options,
            &[b"{\"type\":\"count\",\"id\":0,\"prompt_token_count\":1}\n"],
        )?;
        let mut output = Vec::new();
        let mut client = TokenClient::connect(Cursor::new(input), &mut output, expected(options))?;
        assert_eq!(client.count("ab "), Err(ClientError::RequestTooLarge));
        // JSON escaping counts toward the limit, independently of UTF-8 length.
        assert_eq!(client.count("\"\""), Err(ClientError::RequestTooLarge));
        assert!(client.is_usable());
        assert_eq!(client.count("ab")?, 1);
        let _ = client.into_parts();
        let mut expected_request = request.to_vec();
        expected_request.push(b'\n');
        assert_eq!(output, expected_request);
        Ok(())
    }

    fn assert_fatal(response: &[u8], encode: bool) -> Result<()> {
        let input = script(config(), &[response])?;
        let mut output = Vec::new();
        let mut client = TokenClient::connect(
            BufReader::with_capacity(1, input.as_slice()),
            &mut output,
            expected(config()),
        )?;
        let error = if encode {
            client.encode("private-prompt").err()
        } else {
            client.count("private-prompt").err()
        };
        assert_eq!(error, Some(ClientError::InvalidResponse));
        assert!(!client.is_usable());
        assert_eq!(
            client.count("another-private-prompt"),
            Err(ClientError::Unavailable)
        );
        assert_eq!(
            client.encode("another-private-prompt"),
            Err(ClientError::Unavailable)
        );
        let _ = client.into_parts();
        assert_eq!(std::str::from_utf8(&output)?.lines().count(), 1);
        Ok(())
    }

    #[test]
    fn bad_correlation_shapes_counts_and_worker_errors_permanently_fail_closed() -> Result<()> {
        for response in [
            b"{\"type\":\"count\",\"id\":1,\"prompt_token_count\":1}\n".as_slice(),
            b"{\"type\":\"count\",\"id\":0,\"prompt_token_count\":4}\n",
            b"{\"type\":\"count\",\"id\":0,\"prompt_token_count\":-1}\n",
            b"{\"type\":\"count\",\"id\":0,\"prompt_token_count\":1.0}\n",
            b"{\"type\":\"count\",\"id\":0,\"prompt_token_count\":1,\"extra\":\"private\"}\n",
            b"{\"type\":\"count\",\"id\":0,\"id\":0,\"prompt_token_count\":1}\n",
            b"{\"type\":\"count\",\"type\":\"count\",\"id\":0,\"prompt_token_count\":1}\n",
            b"{\"type\":\"count\",\"id\":0}\n",
            b"{\"type\":\"encoded\",\"id\":0,\"prompt_token_count\":1,\"token_ids\":[1]}\n",
            b"{\"type\":\"error\",\"id\":null,\"code\":\"invalid_request\"}\n",
            b"{\"type\":\"error\",\"id\":0,\"code\":\"invalid_request\"}\n",
            b"{\"type\":\"error\",\"id\":1,\"code\":\"rejected_input\"}\n",
            b"{\"type\":\"error\",\"id\":0,\"code\":\"private-unknown\"}\n",
            b"{\"type\":\"error\",\"code\":\"rejected_input\"}\n",
            b"{\"type\":\"error\",\"id\":0,\"code\":\"rejected_input\",\"private\":true}\n",
            b"{\"type\":\"count\",\"id\":0,\"prompt_token_count\":1} {}\n",
            b"[]\n",
            b"\n",
            b"private-malformed\n",
            b"\xff\n",
            b"{\"type\":\"count\",\"id\":0,\"prompt_token_count\":1}",
            b"",
        ] {
            assert_fatal(response, false)?;
        }
        Ok(())
    }

    #[test]
    fn encoded_responses_require_exact_lengths_and_in_vocabulary_u32_ids() -> Result<()> {
        for response in [
            b"{\"type\":\"encoded\",\"id\":0,\"prompt_token_count\":2,\"token_ids\":[1]}\n".as_slice(),
            b"{\"type\":\"encoded\",\"id\":0,\"prompt_token_count\":4,\"token_ids\":[1,2,3,4]}\n",
            b"{\"type\":\"encoded\",\"id\":0,\"prompt_token_count\":1,\"token_ids\":[260]}\n",
            b"{\"type\":\"encoded\",\"id\":0,\"prompt_token_count\":1,\"token_ids\":[4294967296]}\n",
            b"{\"type\":\"encoded\",\"id\":0,\"prompt_token_count\":1,\"token_ids\":[-1]}\n",
            b"{\"type\":\"encoded\",\"id\":0,\"prompt_token_count\":1,\"token_ids\":[1.0]}\n",
            b"{\"type\":\"encoded\",\"id\":1,\"prompt_token_count\":0,\"token_ids\":[]}\n",
            b"{\"type\":\"encoded\",\"id\":0,\"prompt_token_count\":1,\"token_ids\":[1],\"token_ids\":[1]}\n",
            b"{\"type\":\"count\",\"id\":0,\"prompt_token_count\":1}\n",
        ] {
            assert_fatal(response, true)?;
        }
        Ok(())
    }

    #[test]
    fn response_bounds_include_cr_and_do_not_drain_oversized_frames() -> Result<()> {
        let ready = serde_json::to_vec(&ready(config()))?;
        let limit = ready.len() + 1;
        let mut options = expected(config());
        options.max_response_frame_bytes = limit;
        let mut input = ready.clone();
        input.extend_from_slice(b"\r\n");
        let mut response = b"{\"type\":\"count\",\"id\":0,\"prompt_token_count\":1}".to_vec();
        response.resize(limit - 1, b' ');
        response.extend_from_slice(b"\r\n");
        input.extend_from_slice(&response);
        let mut client = TokenClient::connect(Cursor::new(input), Vec::new(), options)?;
        assert_eq!(client.count("ab")?, 1);

        let mut oversized = ready;
        oversized.push(b'\n');
        let response_start = oversized.len();
        response.insert(response.len() - 1, b' ');
        oversized.extend_from_slice(&response);
        let mut client = TokenClient::connect(
            BufReader::with_capacity(1, Cursor::new(oversized)),
            Vec::new(),
            options,
        )?;
        assert_eq!(client.count("ab"), Err(ClientError::InvalidResponse));
        let (reader, _) = client.into_parts();
        // The fatal oversized frame is not drained or followed by a retry.
        assert!(reader.get_ref().position() < u64::try_from(response_start + response.len())?);
        Ok(())
    }

    struct FailingRead {
        calls: usize,
    }

    impl Read for FailingRead {
        fn read(&mut self, _buffer: &mut [u8]) -> io::Result<usize> {
            self.calls += 1;
            Err(io::Error::other("private-transport-detail"))
        }
    }

    struct FailingWrite {
        fail_flush: bool,
        calls: usize,
    }

    impl Write for FailingWrite {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.calls += 1;
            if self.fail_flush {
                Ok(bytes.len())
            } else {
                Err(io::Error::other("private-transport-detail"))
            }
        }

        fn flush(&mut self) -> io::Result<()> {
            self.calls += 1;
            Err(io::Error::other("private-transport-detail"))
        }
    }

    #[test]
    fn read_write_and_flush_failures_have_static_errors_and_prevent_more_io() -> Result<()> {
        for fail_flush in [false, true] {
            let input = script(config(), &[])?;
            let mut output = FailingWrite {
                fail_flush,
                calls: 0,
            };
            let mut client =
                TokenClient::connect(Cursor::new(input), &mut output, expected(config()))?;
            assert_eq!(client.count("private-prompt"), Err(ClientError::Transport));
            assert_eq!(
                client.count("private-prompt"),
                Err(ClientError::Unavailable)
            );
            assert!(!client.is_usable());
            let _ = client.into_parts();
            assert_eq!(output.calls, if fail_flush { 3 } else { 1 });
        }
        let input = Cursor::new(script(config(), &[])?).chain(FailingRead { calls: 0 });
        let mut client =
            TokenClient::connect(BufReader::new(input), Vec::new(), expected(config()))?;
        let error = client
            .count("private-prompt")
            .err()
            .ok_or("transport failure accepted")?;
        assert_eq!(error, ClientError::Transport);
        assert_eq!(error.to_string(), "token client transport failed");
        assert_eq!(
            client.count("private-prompt"),
            Err(ClientError::Unavailable)
        );
        let (reader, output) = client.into_parts();
        assert_eq!(reader.get_ref().get_ref().1.calls, 1);
        assert_eq!(std::str::from_utf8(&output)?.lines().count(), 1);
        let result = TokenClient::connect(
            BufReader::new(FailingRead { calls: 0 }),
            Vec::new(),
            expected(config()),
        );
        assert!(matches!(result, Err(ClientError::Transport)));
        Ok(())
    }

    fn tokenizer_fixture() -> Result<Vec<u8>> {
        fn text(bytes: &mut Vec<u8>, value: &str) -> Result<()> {
            bytes.extend(u64::try_from(value.len())?.to_le_bytes());
            bytes.extend_from_slice(value.as_bytes());
            Ok(())
        }

        let mut bytes = Vec::new();
        for value in [0x4655_4747_u32, 3] {
            bytes.extend(value.to_le_bytes());
        }
        for value in [1_u64, 8] {
            bytes.extend(value.to_le_bytes());
        }
        for (key, value) in [
            ("general.architecture", "tokenizer-fixture"),
            ("tokenizer.ggml.model", "gpt2"),
            ("tokenizer.ggml.pre", "qwen35"),
        ] {
            text(&mut bytes, key)?;
            bytes.extend(8_u32.to_le_bytes());
            text(&mut bytes, value)?;
        }
        text(&mut bytes, "tokenizer.ggml.tokens")?;
        for value in [9_u32, 8] {
            bytes.extend(value.to_le_bytes());
        }
        bytes.extend(2_u64.to_le_bytes());
        text(&mut bytes, "a")?;
        text(&mut bytes, "<stop>")?;
        text(&mut bytes, "tokenizer.ggml.token_type")?;
        for value in [9_u32, 5] {
            bytes.extend(value.to_le_bytes());
        }
        bytes.extend(2_u64.to_le_bytes());
        for value in [1_i32, 3] {
            bytes.extend(value.to_le_bytes());
        }
        text(&mut bytes, "tokenizer.ggml.merges")?;
        for value in [9_u32, 8] {
            bytes.extend(value.to_le_bytes());
        }
        bytes.extend(0_u64.to_le_bytes());
        text(&mut bytes, "tokenizer.ggml.eos_token_id")?;
        for value in [4_u32, 1] {
            bytes.extend(value.to_le_bytes());
        }
        text(&mut bytes, "tokenizer.ggml.add_eos_token")?;
        bytes.extend(7_u32.to_le_bytes());
        bytes.push(1);
        text(&mut bytes, "token_embd.weight")?;
        bytes.extend(2_u32.to_le_bytes());
        for value in [1_u64, 2] {
            bytes.extend(value.to_le_bytes());
        }
        bytes.extend(0_u32.to_le_bytes());
        bytes.extend(0_u64.to_le_bytes());
        bytes.resize(bytes.len().div_ceil(32) * 32 + 8, 0);
        Ok(bytes)
    }

    #[test]
    fn actual_verified_worker_records_interoperate_with_the_client() -> Result<()> {
        use std::fs;
        use std::time::{SystemTime, UNIX_EPOCH};

        use bonsai_prism_runtime::token_worker::serve;
        use bonsai_prism_runtime::tokenization::TokenizationSession;
        use sha2::{Digest, Sha256};

        let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let root = std::env::temp_dir().join(format!(
            "SwiftBonsai-TokenClient-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir(&root)?;
        let path = root.join("model.gguf");
        let result = (|| -> Result<()> {
            let bytes = tokenizer_fixture()?;
            fs::write(&path, &bytes)?;
            let hash: String = Sha256::digest(&bytes)
                .into_iter()
                .flat_map(|byte| [byte >> 4, byte & 15])
                .map(|nibble| char::from(b"0123456789abcdef"[usize::from(nibble)]))
                .collect();
            let session =
                TokenizationSession::load(&path, u64::try_from(bytes.len())?, &hash, 16 * 1024)?;
            fs::remove_file(&path)?;
            let requests = b"{\"id\":0,\"op\":\"count\",\"text\":\"a\"}\n{\"id\":1,\"op\":\"encode\",\"text\":\"aaaa\"}\n{\"id\":2,\"op\":\"encode\",\"text\":\"aa\"}\n{\"id\":3,\"op\":\"encode\",\"text\":\"\"}\n";
            let mut responses = Vec::new();
            serve(
                &session,
                config(),
                &mut Cursor::new(requests),
                &mut responses,
            )?;
            let mut pinned = expected(config());
            pinned.model_sha256 = &hash;
            pinned.vocab_size = 2;
            let mut emitted = Vec::new();
            let mut client = TokenClient::connect(Cursor::new(responses), &mut emitted, pinned)?;
            assert_eq!(client.count("a")?, 1);
            assert_eq!(client.encode("aaaa"), Err(ClientError::RejectedInput));
            assert_eq!(client.encode("aa")?.token_ids, [0, 0]);
            assert_eq!(client.encode("")?.token_ids, [0_u32; 0]);
            let _ = client.into_parts();
            assert_eq!(emitted, requests);
            for special_input in [SpecialInputPolicy::Literal, SpecialInputPolicy::Parse] {
                let mut options = config();
                options.text_policy.special_input = special_input;
                options.text_policy.insert_special = SpecialTokenInsertion::Model;
                let requests = b"{\"id\":0,\"op\":\"count\",\"text\":\"a\"}\n{\"id\":1,\"op\":\"encode\",\"text\":\"\"}\n{\"id\":2,\"op\":\"count_tokens\",\"token_ids\":[]}\n{\"id\":3,\"op\":\"encode_tokens\",\"token_ids\":[1,0,1]}\n{\"id\":4,\"op\":\"encode_tokens\",\"token_ids\":[]}\n{\"id\":5,\"op\":\"count_tokens\",\"token_ids\":[1,1]}\n";
                let mut responses = Vec::new();
                serve(
                    &session,
                    options,
                    &mut Cursor::new(requests),
                    &mut responses,
                )?;
                let mut pinned = expected(options);
                pinned.model_sha256 = &hash;
                pinned.vocab_size = 2;
                let mut emitted = Vec::new();
                let mut client =
                    TokenClient::connect(Cursor::new(responses), &mut emitted, pinned)?;
                // Text receives the model's EOS; raw IDs bypass insertion.
                assert_eq!(client.count("a")?, 2);
                assert_eq!(client.encode("")?.token_ids, [1]);
                assert_eq!(client.count_tokens(&[])?, 0);
                assert_eq!(client.encode_tokens(&[1, 0, 1])?.token_ids, [1, 0, 1]);
                assert_eq!(client.encode_tokens(&[])?.token_ids, [0_u32; 0]);
                assert_eq!(client.count_tokens(&[1, 1])?, 2);
                let _ = client.into_parts();
                assert_eq!(emitted, requests);
            }
            Ok(())
        })();
        // Remove only paths created by this fixture, including after failure.
        let _ = fs::remove_file(path);
        fs::remove_dir(root)?;
        result
    }
}
