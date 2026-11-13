# CLI and workers

[Documentation index](index.md) · [Build and validation](build-and-validation.md) · [Runtime contracts](runtime-contracts.md)

Use each executable's `--help` for its complete startup arguments.

## Inspect compiled support

Run `prism-inference --capabilities` alone for a version-one JSON record without
model I/O or device execution. `runtime.target` describes the executable target,
including emulation. `cpu.neon_build_eligible` reports build eligibility, not host
ISA detection or per-operation dispatch. It is false on unsupported targets and
without `experimental-neon`.

`supported_contracts` lists generic Qwen35 and pinned Swift graph-identity
suffixes; complete decoder identities also include the verified model SHA-256.
These identify arithmetic contracts, not portable checkpoints. Vulkan and Hexagon
are unavailable through this entry point; `qualification` is `unqualified`.
Consumers must reject unsupported schema versions and independently verify
artifacts and resource admission.

## Generate from text or token IDs

Generation defaults to `--mode generate`. It requires an absolute model path,
expected length/SHA-256, retained-model and context/checkpoint budgets, one input
form and a generation limit. Bounds and identity syntax are checked before model
I/O; vocabulary IDs are checked after verified loading.

| Input | Format |
| --- | --- |
| `--prompt <text>` | Plain UTF-8 text |
| `--prompt-file <absolute-path\|->` | Exact UTF-8 from a file or stdin through EOF |
| `--tokens <id,id,...>` | Exact token IDs |
| `--tokens-file <absolute-path\|->` | One strict JSON array of unsigned 32-bit IDs |

Inputs are mutually exclusive and must be nonempty. File/stdin input has a 1 MiB
byte bound, independent of the prompt-token allowance. Text preserves CR/LF and
a UTF-8 BOM; malformed UTF-8 rejects. Token arrays preserve order and duplicates;
comments, trailing JSON, negative/fractional IDs and values above `4294967295`
reject. File/stdin reads have no timeout; other option validation precedes them.

Text encoding applies no chat template. `--special-input literal|parse` defaults
to literal. `--insert-special none|model` defaults to none; model mode applies the
verified BOS/EOS insertion flags without suppressing existing boundary IDs.
Inserted IDs consume the prompt allowance. Token-ID input accepts neither option
and applies no parsing or insertion.

The encoded prompt must leave the full output allowance in context. The model
must declare a valid EOS ID; EOS and optional EOT/EOM always stop generation.
`--stop-tokens` adds IDs under a combined 256-distinct-ID bound. Stops are neither
emitted nor committed. Completion accounting records the effective input and stop
policies.

## Streaming output and generation controls

The exact verified Swift PQ2/PTQ1 identities default to K4/V4 `--kv-cache rsq4`;
other models use FP16. Rotated profiles require 256-channel full-attention heads
and reject unsupported geometry. Select `f16` for reference/recovery or opt into
`rsq4k3v`; see [quantized KV](quantized-kv.md).

Output is one JSON token object per line followed by accounting. `--output text`
adds `bytes`, `text` and `special`, requiring matching `gpt2`/`qwen35` vocabulary
metadata. `bytes` preserves the token exactly. `text` buffers incomplete UTF-8
across records and replaces invalid sequences with U+FFFD. Control-token spellings
remain literal and carry `special: true`. Final accounting flushes any incomplete
suffix. Concatenate all `text` fields for display or all token `bytes` for
lossless output. All output is provisional until final success.

| Option | Behavior |
| --- | --- |
| `--timeout-seconds <1..3600>` | Bounds the runtime request, including prefill, delivery and checkpoints; excludes model loading and graph construction. Omission leaves cancellation-only behavior. |
| `--prefill-block-tokens <1..256>` | Reference block boundary, default one; graph operations remain sequential. |
| `--sampling reference --seed <u64>` | Experimental stochastic mode; greedy is the default. |
| `--temperature`, `--top-k`, `--top-p` | Require reference sampling; defaults 1, 20 and 0.95. Top-k zero keeps the full vocabulary; top-p is in `(0, 1]`. |

A timeout fails without a success record. The completion records the selected
cache, graph, prefill boundary and sampling settings. Reference sampling draws
the upper 32 SplitMix64 bits after ranking, temperature, top-k and nucleus
selection; it does not reproduce Prism's C++ RNG.

## Sampling and stop policies

Library callers supply an independently cloneable `Sampler` to
`Runtime::generate_sampled` or `generate_sampled_until`. Errors restore sampler
and graph state, even if decoder recovery fails. Successful stop-token draws
advance the sampler. External side effects and shared mutable sampler state are
outside this rollback contract.

`SpecialTokens::generation(max_tokens, caller_stops)` builds the same checked
EOS/EOT/EOM-plus-caller stop set as the driver, preserving first occurrence order.
It validates IDs, the 256-stop bound and space for a nonempty prompt. Insertion
flags do not disable generation stops.

## Tokenize without constructing the graph

`--mode tokenize` requires `--model`, `--bytes`, `--sha256`, `--memory-bytes`,
`--context` and one input form from the table above. `--max-tokens` reserves future
output slots and defaults to zero. Text uses the same parsing/insertion policies
as generation. Generation-only options, including `--checkpoint-bytes`, reject.

One JSON record reports IDs, count, context, reserved output, input policies and
artifact identity. Verification reads the whole file, but the command retains
only bounded metadata and constructs no inference graph. The metadata budget
excludes tokenizer scratch and output buffers; it is not a process-memory cap.

## Reusable tokenization and text sessions

Retain `tokenization::TokenizationSession` for repeated requests.
`TextTokenizationPolicy` selects special-input handling, boundary insertion and
`TokenBudget { context, reserved_output_tokens }`.

| Method | Result |
| --- | --- |
| `encode` | Text IDs and prompt/context/output accounting |
| `check_tokens` | Validated copy of raw IDs, with no insertion |
| `validate_tokens` | Same raw-ID checks on borrowed storage, without allocation |

Construction verifies the artifact and complete tokenizer, then releases the file
handle and metadata. Invalid requests leave the session reusable. Raw CLI input
can use metadata alone and does not require BPE/pretokenizer support.

`text_session::TextOutputSession` separately verifies the artifact and retains
only vocabulary data. `piece(id)` borrows exact bytes; `is_control(id)` reports
the declared type. Feed pieces to a request-local `Utf8Output`, finish on success
and reset abandoned output. Vocabulary count, per-piece size and decoded-byte
bounds are independent of whole-process memory.

## Persistent token worker

`bonsai-token-worker` requires absolute `--model`, verified `--bytes`/`--sha256`,
`--memory-bytes`, `--context`, `--reserved-output-tokens`,
`--special-input literal|parse`, `--insert-special none|model` and
`--max-frame-bytes` (1 through 1,048,576). Run from an external system-temp working
directory; see [building](build-and-validation.md#build-a-checkout).

Wait for the flushed `type: "ready"` record. It identifies protocol
`bonsai-token-worker-v1`, verified model SHA-256, vocabulary and fixed settings.
Requests are LF-terminated JSON objects:

```json
{"id":1,"op":"count","text":"Draft a scene."}
{"id":2,"op":"encode","text":"Draft a scene."}
{"id":3,"op":"count_tokens","token_ids":[42,17]}
{"id":4,"op":"encode_tokens","token_ids":[42,17]}
```

Responses echo the ID and `prompt_token_count`; `type: "encoded"` adds ordered
`token_ids`, while `type: "count"` omits them. Raw operations preserve exact IDs,
including empty arrays, with no parsing/insertion. Every operation reserves the
startup output allowance. `text` and `token_ids` are exclusive and must match the
operation. IDs are correlation values without a global registry.

Duplicate/unknown fields and malformed frames return `invalid_request` with a
null ID. Encoding/context rejection returns `rejected_input` with the parsed ID.
Both preserve the session and omit prompt text and internal diagnostics.

The frame bound counts bytes before LF, including CR in CRLF. An unterminated
final frame receives `invalid_request` and clean exit. Oversized frames and
transport/allocation errors terminate without draining input: discard incomplete
output and restart. The synchronous transport has no deadline or cancellation
protocol.

## Token worker client

`token_client::TokenClient::connect` takes caller-owned buffered streams and
`ExpectedWorker`: pinned lowercase model SHA-256, vocabulary size, exact
`WorkerConfig` and a response-frame bound. Readiness must match before requests.

`count`/`encode` handle text; `count_tokens`/`encode_tokens` preserve exact raw
sequences. Invalid IDs or prompt overflow reject locally before writing or
consuming an ID. Raw counts must equal the submitted length; raw encode replies
must match every ID. Request and response bounds independently range from 1 byte
through 1 MiB; encoded responses can exceed request size.

Malformed, oversized, uncorrelated or inconsistent responses and transport errors
disable the client. Local size rejection sends nothing; correlated
`rejected_input` leaves the connection reusable. Errors use static diagnostics.
The caller owns deadlines, cancellation, process lifetime and shutdown.
`into_parts` returns the streams without clearing a fault.

## Input readers and tokenizer APIs

`prompt_input::read_prompt` reads at most 1 MiB plus one overflow byte and
preserves exact UTF-8. Empty, oversized, malformed or failed reads return no
partial text. I/O errors retain their kind with sanitized messages, including
interruption. Callers own deadlines and must rewind or replace a failed stream.

`token_input::read_tokens` applies a token limit and the same independent byte
bound to one strict unsigned-32-bit JSON array. The library accepts an empty
array; the driver requires a nonempty prompt. Failed reads return no partial IDs.

`Bpe::from_metadata` validates vocabulary and ranked merges.
`encode_piece` takes pretokenized bytes; `encode_pieces` preserves supplied
boundaries. Use `Tokenizer::encode` to apply Qwen35 boundaries and special-input
policy, or `encode_with_special_tokens` with a validated same-model
`SpecialTokens` contract for insertion. These APIs do not render chat templates.

`Qwen35PreTokenizer::pieces` borrows text and stops on error; consume the complete
iterator to validate all piece limits. `split` collects the pieces. Request-local
encoding scratch is bounded by the largest admitted piece.

## Swift worker and private HTTP server

`bonsai-swift-worker` serves the pinned Swift model over JSONL.
`SwiftClient` checks readiness before count/completion/reset exchanges.
`SwiftChatSession` applies the verified template and tokenizer; `SwiftSession`
provides the MT19937 sampler and transactional decoding. See
[runtime contracts](runtime-contracts.md#status-and-integration) and
[session profiles](swift-session-profiles.md).

The worker defaults to K4/V4 and accepts trusted startup
`--kv-cache f16|rsq4|rsq4k3v`; requests cannot change it. Non-FP16 readiness adds
`kv_cache` and `startup_graph_identity`. `SwiftClient::connect` checks the default;
`connect_with_kv_cache` checks an explicit profile. The caller owns transport
deadlines, cancellation and process lifetime.

`prism-inference-server` provides the private authenticated HTTP contract for an
owning launcher. It also defaults to K4/V4 with explicit `f16` recovery. The owner
verifies executable/model identity, readiness and process/Job ownership. See the
[server guide](../server/README.md).
