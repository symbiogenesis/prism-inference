# CLI and workers

[Documentation index](index.md) · [Build and validation](build-and-validation.md) · [Runtime contracts](runtime-contracts.md)

## Inspect compiled support

Run `prism-inference --capabilities` alone to emit one bounded version-one JSON
record without model I/O or device execution. The `runtime.target` facts describe
the executable target, including an emulated process. `cpu.neon_build_eligible`
uses the same production kernel predicate as actual dispatch; it is a build fact,
not a host ISA probe or a promise that every operation uses NEON. Unsupported
targets and builds without the `experimental-neon` feature report false.

`supported_contracts` lists the graph identity suffixes used by the generic
Qwen35 decoder and the exact pinned Swift artifact. Complete decoder identities
also include the verified model SHA-256. These suffixes identify implementation
contracts; they do not establish portable checkpoints or qualify a model/device.
Vulkan and Hexagon are unavailable and `qualification` remains `unqualified`.
Consumers must reject unsupported schema versions and retain independent artifact,
evidence and resource admission. This inventory does not select a profile or
change generation defaults. Additional options alongside `--capabilities` reject.

## Generate from text or token IDs

Generation (the default `--mode generate`) requires an absolute model path,
expected length/SHA-256, retained-model and context/checkpoint budgets, one input
form and a generation limit. It validates
identity syntax, positive bounds, prompt-plus-generation capacity and the
256-stop-token limit before model I/O; vocabulary checks follow verified loading.
Supply `--tokens <id,id,...>` from the reference tokenizer, or use experimental
`--prompt <text>` for plain UTF-8 encoding. `--prompt-file <absolute-path>` reads
exact UTF-8 from a file, and `--prompt-file -` reads stdin through EOF. These
input options and `--tokens-file <absolute-path|->` are mutually exclusive.
Token files contain one strict JSON array of unsigned 32-bit IDs, for example
`[42,17,42]`. Whitespace is allowed; comments, trailing JSON, negative/fractional
IDs and values above `4294967295` reject. The driver requires a nonempty array
within the prompt allowance and preserves exact order and duplicates. A token
file has the same 1 MiB byte bound, including whitespace, and no text parsing or
BOS/EOS insertion applies. The byte bound is independent of the token allowance;
a sequence that fits the context can still exceed the JSON byte bound.
Vocabulary validation follows verified model loading.
File/stdin input keeps prompt bodies out of process arguments. Text prompts
preserve every byte, including CR/LF and a UTF-8 BOM; empty, malformed UTF-8 and
inputs larger than 1 MiB reject before model verification. Other option
validation precedes stdin reading; input reads have no timeout. The generation
timeout still begins after model loading.
`--special-input literal|parse` is available only with text input;
literal is the default. Text encoding applies no chat template or BOS/EOS
insertion by default, even if insertion flags are present in metadata. Supply
explicitly formatted text when control tokens are intended. `--insert-special model` opts
text input into the verified model's BOS/EOS insertion flags; `none` is the
default. Only true flags insert IDs. Insertion does not suppress existing
boundary IDs, and all inserted IDs consume the prompt allowance. Token-ID input
rejects this option and remains exact. Text input records
`input_mode: "text"` and the effective `special_input` policy in the completion.
Its exact encoded count must leave the entire output allowance in the context;
overflow rejects before graph execution. The verified model must
declare a valid EOS ID; its EOS and optional EOT/EOM IDs always stop generation.
`--stop-tokens` adds IDs to that set, deduplicated under a combined 256-ID bound.
Stops are neither emitted nor committed to the conversation. BOS/padding IDs and
insertion flags are validated; supplied token-ID prompts remain unchanged.

## Streaming output and generation controls

Output defaults to one JSON token object per line followed by accounting, which
includes the effective `stop_tokens`. Optional `--output text` adds `bytes`,
`text` and `special` to each token record. It requires matching `gpt2`/`qwen35`
vocabulary metadata. `bytes` preserves the decoded token exactly; `text` buffers
incomplete UTF-8 across records and displays invalid byte sequences as U+FFFD.
Literal control-token spellings are retained and marked `special: true`.
The final accounting record has a `text` field that flushes any truncated UTF-8
suffix as U+FFFD; concatenate all `text` fields for display, or all token `bytes`
for lossless output. These records remain provisional until final success.
Optional `--timeout-seconds <1..3600>` bounds generation from immediately before
the runtime request, including prefill, delivery and checkpoint work. Model
verification/loading and graph construction precede that deadline. Omit the
option to retain cancellation-only behavior. A timeout fails the command;
already streamed tokens are provisional and no success record follows.
Optional `--prefill-block-tokens <1..256>` selects the reference block boundary;
the default is one. `prefill_block_tokens` records the selected size. This changes
scheduling boundaries while retaining sequential graph operations.
Optional `--sampling reference --seed <u64>` selects experimental stochastic
sampling. `--temperature`, `--top-k` and `--top-p` default to 1, 20 and 0.95;
top-k zero retains the full vocabulary, and top-p must be in `(0, 1]`.
All sampling controls require that explicit mode. The completion's `sampling`
object records the mode, effective options, seed and RNG. The reference draws
the upper 32 SplitMix64 bits after ranking, temperature, top-k and nucleus
selection; it does not reproduce Prism's C++ RNG or establish stochastic parity.
Greedy remains the default and records only its mode. Model stops still apply.

## Sampling and stop policies

Library callers can pass an independently cloneable `Sampler` through
`Runtime::generate_sampled` or `generate_sampled_until`. Any returned error
restores the sampler alongside graph state, even when decoder recovery fails.
Successful stop-token draws advance the sampler without emitting the stop.
Shared mutable state and external side effects are outside this recovery contract.
`SpecialTokens::generation(max_tokens, caller_stops)` builds the driver's same
checked EOS/EOT/EOM-plus-caller stop set. It validates vocabulary IDs, preserves
first occurrence order and admits at most 256 distinct stops. Output must leave
room for a nonempty prompt within the maximum context; the runtime still checks
the actual prompt/context. Insertion flags do not disable declared generation stops.
The driver does not render chat templates, launch the broker or promote a profile.

## Tokenize without constructing the graph

Use `--mode tokenize` to inspect exact input IDs/count without graph allocation.
This mode requires `--model`, `--bytes`, `--sha256`, `--memory-bytes`, `--context`
and exactly one nonempty `--prompt`, `--prompt-file`, `--tokens` or `--tokens-file`
input. Optional `--max-tokens` reserves that many context slots for future output;
its default is zero. Text input uses the
same special-token and insertion policies as generation. Generation-only options,
including `--checkpoint-bytes`, are rejected. A single JSON record reports
`token_ids`, `token_count`, `context`, `reserved_output_tokens`, effective input
policies and artifact identity. Length/SHA-256 verification still reads the
whole file, and bounded header buffering can read ahead into tensor data. The
command retains only bounded metadata, materializes no tensors and executes no
inference. The metadata budget excludes tokenizer scratch
and output buffers; this is not process-memory admission or broker integration.

## Reusable tokenization and text sessions

Library callers can retain `tokenization::TokenizationSession` across requests.
`TextTokenizationPolicy` makes literal/parse handling, boundary insertion and
`TokenBudget { context, reserved_output_tokens }` explicit. `encode` returns
exact `token_ids` and their prompt/context/output accounting; `check_tokens`
validates and copies raw IDs with the same budget and never inserts tokens.
`validate_tokens` applies those checks to borrowed IDs without allocating, for
callers that only need a count or can serialize their existing storage. The session
builds its tokenizer and special declarations from the same verified artifact,
then releases the metadata and file handle. Invalid requests leave it reusable.
Session construction requires a supported complete tokenizer even for later
raw-ID requests. The CLI's existing raw-ID mode remains metadata-only and does
not require BPE or pretokenizer support. Persistent broker transport and template
rendering remain separate work.

Service callers can similarly load `text_session::TextOutputSession` once with
an independently pinned artifact length/hash and metadata budget. `piece(id)`
borrows exact bytes; `is_control(id)` reports the declared type without choosing
stop or display policy. The session retains no weights, graph, model handle or
request state. Its vocabulary has separate count/per-piece/decoded-byte bounds;
the metadata budget does not bound process memory. Feed each request's pieces
to its own `Utf8Output`, preserving original bytes for lossless output, finishing
on success and resetting abandoned output on cancellation. Neither session
construction nor these synthetic checks establish full-model parity.

## Persistent token worker

The default `cli` feature also builds `bonsai-token-worker`; `--help` lists its
startup contract. Every setting is required: absolute `--model`, verified
`--bytes`/`--sha256`, `--memory-bytes`, `--context`, `--reserved-output-tokens`,
`--special-input literal|parse`, `--insert-special none|model` and
`--max-frame-bytes` (1 through 1,048,576). Run it from an explicit external
system-temp working directory, using the [checkout build commands](build-and-validation.md#build-a-checkout). `cargo run` still selects the original token driver by default.

Read the flushed `type: "ready"` record before sending input. It identifies
protocol `bonsai-token-worker-v1`, the verified model SHA-256, vocabulary and all
fixed text/context/transport settings. Each request is one LF-terminated JSON
object with exactly these fields:

```json
{"id":1,"op":"count","text":"Draft a scene."}
{"id":2,"op":"encode","text":"Draft a scene."}
{"id":3,"op":"count_tokens","token_ids":[42,17]}
{"id":4,"op":"encode_tokens","token_ids":[42,17]}
```

Responses carry the request ID and exact `prompt_token_count`; `type: "encoded"`
also returns ordered `token_ids`, while `type: "count"` omits them. Empty text
follows the library's counting/insertion contract. Raw operations validate every
ID against the verified vocabulary and preserve the exact sequence, including
duplicates, boundary IDs and empty arrays. They apply no parsing or insertion;
their count still reserves the complete startup output allowance. `text` and
`token_ids` are exclusive and must match the selected operation.
IDs are correlation values; the worker retains no global ID registry.
Raw operations validate and reuse the decoded request array without making a
second ID vector. Unescaped text borrows the frame; escaped JSON strings decode
to owned text with the same tokenization policy.
Duplicate/unknown fields and malformed
frames return `invalid_request` with a null ID; encoding/context rejection returns
`rejected_input` with the parsed ID. These failures preserve the session for later
requests and expose no prompt text or underlying diagnostics.

The frame bound counts bytes before LF, including CR in CRLF. An unterminated
final frame receives `invalid_request`, followed by clean exit. Oversized frames
and transport/allocation errors terminate the worker without draining input;
discard incomplete output and restart. The synchronous transport has no deadline
or cancellation protocol. Readiness establishes a verified tokenizer, not a
rendered chat template, broker endpoint, inference backend or daily-service gate.

## Token worker client

With `cli` enabled, `token_client::TokenClient::connect` accepts caller-owned
buffered input/output and an `ExpectedWorker`: independently pinned lowercase
model SHA-256, vocabulary size, exact `WorkerConfig` and a separate response-frame
bound. Readiness must match every identity/policy field before requests are sent.
`count(text)` returns the checked count; `encode(text)` returns checked ordered
IDs and fixed context/output accounting. `count_tokens(ids)` and
`encode_tokens(ids)` preserve raw sequences, including empty input, independently
of the pinned text parsing/insertion policy. Invalid vocabulary IDs or an exceeded
prompt allowance reject locally before writing or consuming a correlation ID.
Raw counts must equal the submitted length, and raw encode replies must match
every submitted ID in order; a mismatch permanently disables the client.
Request and response bounds each range from 1 byte through 1 MiB; an encoded
response can exceed its request size, so choose the response budget explicitly.
Malformed, oversized, uncorrelated or inconsistent responses and transport errors
disable the client. Local request-size rejection sends nothing, and a correlated
`rejected_input` leaves the connection reusable. Errors contain static diagnostics.
The caller owns transport deadlines, cancellation, process lifetime and shutdown;
`into_parts` returns its streams without claiming a faulted connection is reusable.

## Input readers and tokenizer APIs

`prompt_input::read_prompt` is the driver's public file/stream reader. It preserves
valid UTF-8 exactly, consumes at most 1 MiB plus one overflow byte and rejects
empty, oversized or malformed input without returning partial text. I/O errors
retain their kind with sanitized messages and return immediately, including
interruption. Callers own input deadlines and must rewind or replace a failed
stream before retrying a whole prompt.

`token_input::read_tokens` similarly reads one strict JSON array with an explicit
token limit and the independent 1 MiB byte bound. It consumes at most one overflow
byte, uses checked unsigned decimal parsing and returns no partial sequence on
failure. The library accepts an empty array; the driver requires a nonempty
prompt. Neither reading path supplies cancellation or an input deadline.

`Bpe::from_metadata` validates normal vocabulary pieces and ranked merge rules.
Input-only construction checks the full vocabulary without materializing decoded
output pieces; composed tokenizers reuse that checked view for special spellings.
`encode_piece` accepts one already pretokenized byte slice; `encode_pieces`
preserves caller-supplied boundaries. Neither API finds Qwen35 regex boundaries,
recognizes special input tokens, inserts BOS/EOS or renders a chat template.
`Tokenizer::encode` composes those boundaries and the special-input policy for
plain text. `encode_with_special_tokens` additionally applies a validated
`SpecialTokens` contract with the same vocabulary size and explicit insertion
flags; the caller must supply metadata from the same model. Using the merge core
directly on arbitrary text does not establish tokenizer parity.

Multi-piece encoding reuses one request-local node vector and candidate heap;
scratch capacity is bounded by the largest admitted piece. The tokenizer consumes
`Qwen35PreTokenizer::pieces` directly without collecting a slice vector. This
checked iterator borrows the original text and stops after an error; callers must
consume it completely to validate all piece limits. The collecting `split` API
remains available. These changes retain immutable shared tokenizer state and do
not establish measured allocation counts, throughput or full-model parity.

## Swift worker and private HTTP server

`bonsai-swift-worker` serves the currently pinned Swift model over JSONL.
`SwiftClient` verifies readiness before count/completion/reset exchanges; the
caller owns process lifetime, transport deadlines, cancellation and shutdown.
`SwiftChatSession` applies the verified template and tokenizer, while
`SwiftSession` supplies the production MT19937 sampler and transactional
decoding. See [runtime contracts](runtime-contracts.md#status-and-integration).

`prism-inference-server` supplies the private authenticated HTTP contract used
by SwiftBonsai's guarded Windows launcher. The owner verifies executable and
model identity and retains process/Job ownership. See the
[server guide](../server/README.md) for its ownership and streaming contract.
Use each executable's `--help` for its startup arguments.
