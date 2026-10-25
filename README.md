# Prism Inference

`prism-inference` is an experimental safe Rust CPU inference runtime for
Prism-compatible GGUF models, extracted from SwiftBonsai into an independent
repository. Its first implemented generation path is the dense Qwen35 text
architecture used by Swift/Bonsai; broader model support remains future work.
It links no C/C++ inference library.
**Full-model numerical parity and complete device lifecycle qualification remain
pending.** The experimental Swift session and owned broker integration exist;
pinned publisher Prism remains the assessment and fallback engine.

The package provides the `prism-inference` token driver, `bonsai-token-worker`
metadata service and `bonsai-swift-worker` inference service. The worker names,
the Swift readiness identity `bonsai-prism-runtime` and the Rust library import
`bonsai_prism_runtime` remain compatible with SwiftBonsai for this first release.
This is an independent implementation, not an official PrismML release.

The [MVP checklist](https://github.com/symbiogenesis/SwiftBonsai2/blob/370084c530e2271f10a1c55a5a10bba7c806aacb/docs/prism-upstream-todo.md) separates implemented
behavior from measured qualification. The actual pinned 851-tensor model loads
and its 8K graph initializes within the CPU budget. Scalar/NEON state and logits
match exactly on the retained three-step trajectory, while native numerical
differences exceed the declared tolerance. Do not promote this engine on the
basis of loading, matching sampled observations or a single streamed response.

`SwiftChatSession` applies the verified template and exact tokenizer with
thinking on, `xhigh`, preserved supplied reasoning and complete output
reservation. `SwiftSession` joins it to the fixed production MT19937 sampler,
all five verified native stop tokens, transactional decoding and bounded UTF-8
output. The separate `bonsai-swift-worker` JSONL executable and `SwiftClient`
verify readiness and bounded count/completion/reset exchanges. Their caller
owns process lifetime and transport shutdown.

SwiftBonsai consumes the released `prism-inference-server` executable inside its
existing guarded Windows Job. The private HTTP server preserves authentication,
exact prepared-input binding, bounded reasoning/content streaming and reset
requirements. Ordinary SwiftBonsai builds do not compile this runtime.

The release also includes `prism-inference`, `bonsai-token-worker`,
`bonsai-swift-worker` and `bonsai-gguf`. The legacy worker names and Rust library
name `bonsai_prism_runtime` preserve first-release protocol/API compatibility.
The crates.io package and repository are both named `prism-inference`.

Broader Prism architectures and general model selection are in scope and deferred;
the implemented model-specific Swift path remains experimental. See the
[upstream roadmap](https://github.com/symbiogenesis/SwiftBonsai2/blob/370084c530e2271f10a1c55a5a10bba7c806aacb/docs/prism-upstream-todo.md)
for qualification that remains open.

## Implemented behavior

- Verified bounded GGUF loading through `bonsai-gguf`: file length/SHA-256 are
  checked before parsing, and payloads use the same retained handle. Windows
  denies write/delete sharing; other platforms require caller-enforced immutability.
- Immutable F32/F16/BF16/PQ2/PTQ1 tensors, checked dimensions, retained-storage
  budgets and optional byte-identical rank-one LoRA B sharing. Storage sharing
  does not implement adapter forward execution.
- Packed scalar matrix operations with PQ2 Q8_K and PTQ1 Q8_0 activation
  contracts, plus automatic baseline NEON on eligible Windows ARM64 operations.
  Scalar remains the reference/fallback; no expanded float matrix is retained.
  Synthetic integer/matvec parity does not establish whole-model parity or speed.
- Embeddings, Hadamard/sign transforms, RMS normalization, causal nonflash
  attention, text-interleaved MRoPE, GDN/convolution state, gated feed-forward
  layers and output projection. Attention retains FP16 query/probability
  conversion and exact FP16 KV bits; convolution/GDN state stays F32.
- Single-owner bounded conversations with greedy token streaming, backpressure,
  exact prefix checkpoints, edited prompts, reset, cancellation between decoder
  operations and rollback after evaluation or delivery failure. Streamed tokens
  are provisional until success. Failed restore preserves the public committed
  token prefix and closes admission until reset succeeds.
- Explicit seeded reference sampling with temperature, top-k and top-p, stable
  token-ID tie breaking and a documented SplitMix64 draw. Caller-owned samplers
  participate in request rollback, including cancellation, timeout and failed
  delivery. Greedy remains the default; Prism RNG and stochastic parity remain open.
- `Runtime::generate_until` accepts an absolute monotonic deadline and rolls
  back expiry through the same transaction. Checks surround decoder operations
  and stream callbacks; synchronous work must return before expiry is detected,
  and rollback itself must complete before the method returns.
- `Decoder::evaluate_into` reuses caller-owned logits; `evaluate` remains
  available. Qwen35 retains bounded graph/kernel scratch and stages per-layer
  state deltas. Outputs are shape/finiteness checked; errors clear direct output.
  Checkpoints own independent committed state/logits, never scratch aliases.
- `Decoder::ingest` skips output-only normalization/projection for intermediate
  uncached prompt tokens while validating computed state. Its compatibility
  fallback computes/validates then discards logits. Final prompt tokens and
  decode steps produce logits; exact prefix hits reuse saved output.
- `Decoder::ingest_block` provides bounded 1–256-token reference prefill.
  `Runtime::generate_with_prefill` and `generate_until_with_prefill` accept
  `Prefill { block_tokens }`; existing generation calls retain one-token blocks.
  The default block decoder preserves exact sequential arithmetic and checks
  cancellation/deadlines between tokens. Errors roll back the whole request.
  Optimized tiled graph execution and prefill performance remain unqualified.
- Full-attention checkpoints share sealed 64-token FP16 KV pages. Each snapshot
  copies the private mutable tail and complete F32 convolution/recurrent state;
  later append, reset or restoration cannot modify saved pages.
- Scoped prepared matvecs share activation preparation across compatible FFN,
  full-attention and recurrent sibling projections. Groups require the same
  exact input, transform/permutation, width and activation contract; mixed
  contracts prepare separately. Prepared storage cannot outlive its borrowed
  input or scratch. Exact synthetic checks do not establish device speed gains.
- Optional checkpoint replacement preserves older prefixes when capture fails
  or the replacement alone exceeds the available snapshot budget. Count-based
  eviction follows successful capture; byte-pressure eviction still precedes
  allocation to keep the hard snapshot bound.
- Failed decoder resets keep the public committed history and saved prefixes
  while closing admission. A successful reset clears them and restores admission.
- Validated model-declared EOS/EOT/EOM stops, combined with additional caller
  stops in the token driver. BOS/padding IDs and insertion flags are checked;
  supplied prompt token IDs receive no automatic token insertion.
- Bounded `gpt2`/`qwen35` output vocabulary and optional incremental text rendering.
  Exact token bytes remain available across UTF-8 token boundaries; control-token
  spellings remain literal. Chat templates remain separate.
- Ranked byte-pair merges for already pretokenized GPT-2 byte pieces. The bounded
  merge core emits only normal IDs and preserves supplied piece boundaries.
- Bounded plain-text input with the pinned Prism Qwen35 scanner and Unicode
  tables, explicit literal/parse special-token handling and ranked BPE. Optional
  model-declared BOS/EOS insertion is explicit and shares the prompt token bound.
  Raw encoding remains the default. Chat templates and full-model tokenizer
  parity remain open.
- Verified tokenization-only mode returns exact prompt IDs and counts from
  bounded metadata without retaining weights or constructing the graph.
- Reusable `TokenizationSession` verifies the artifact once, retains its
  tokenizer and special declarations, and shares explicit context/output
  accounting across text and checked raw-ID requests. Metadata and the artifact
  handle are released after construction; requests require no model file I/O.
- Public `text_output::Utf8Output` renders bounded caller-owned byte chunks,
  retains at most three incomplete UTF-8 bytes, and resets abandoned output
  before a new request. The token driver uses this same decoder.
- Reusable `text_session::TextOutputSession` verifies one output vocabulary,
  releases model metadata and the file handle, and supplies borrowed lossless
  pieces and declared control status to independent caller-owned display streams.
- A persistent `bonsai-token-worker` verifies one tokenizer artifact at startup,
  flushes readiness and serves bounded JSONL count/encode requests without graph
  allocation or subsequent artifact I/O. Text and exact raw-ID operations share
  its fixed context/output budget; parsing and insertion apply only to text.
- A bounded synchronous `token_client::TokenClient` checks worker readiness
  against pinned identity/policy before text or raw-ID count/encode exchanges.
  Strict framing, response schemas and correlation fail closed after transport faults.
- Library callers share the driver's bounded UTF-8 prompt and JSON token-array
  readers and checked model-plus-caller generation-stop policy.
- `jinja::Template::chat_capabilities` probes tool definitions, tool calls, system
  roles, parallel tool calls, typed content and string content using the pinned
  Prism markers, usage observations and exception/fallback rules. The existing
  content-only API remains available. These probes do not qualify general chat
  conversion or template behavior for every input.
- `codec::CodecBuffers` owns bounded persistent byte buffers with checked regions,
  monotonic handles and explicit reference/CPU/decode/dot calls. Unaligned and
  overlapping regions use safe byte access, and failed calls preserve output.
  The complete pinned 144-slot trait inventory distinguishes native function
  availability from implemented Rust operations; unavailable CPU quantizers
  reject instead of using a reference implementation. Borrowed row decoding
  reuses existing kernels without copying input into an owned tensor.

The graph/quantizer reference is Prism
`73bfe6cc485554822f80c473a3c0c4f25e80f41b`, matching the assessment runtime.
The source-overlay build lock is a separate identity. Audited source paths/hashes
are in [kernels.rs](src/kernels.rs); MIT terms are in [NOTICE](NOTICE) and
[PRISM-LICENSE.txt](PRISM-LICENSE.txt). HTTP framing and Q1_0/Q2_0 decoder
adaptations also retain Camelid's MIT terms in
[CAMELID-LICENSE.txt](CAMELID-LICENSE.txt).

## Build and use

Install the published source package with the pinned nightly:

```powershell
cargo +nightly-2026-10-22 install prism-inference --version 0.1.0 --locked
prism-inference --help
```

`cargo install` compiles source. Consumers using a prebuilt binary bundle should
pin its version, target and SHA-256 rather than rebuild the runtime on startup.
Library consumers can retain the existing import path with this dependency:

```toml
bonsai-prism-runtime = { package = "prism-inference", version = "=0.1.0" }
```

Default features are `cli` (JSON token driver) and `experimental-neon`.

| Cargo feature selection | Result |
| --- | --- |
| Defaults | Token driver/library with eligible NEON selection |
| `--no-default-features --features experimental-neon --lib` | NEON-enabled library without JSON driver |
| `--no-default-features --lib` | Scalar-only library |
| `--no-default-features --features cli` | Scalar-only token driver |

Use the pinned Rust nightly-2026-10-22 and an absolute manifest/target path from
a unique system-temp working directory. After fetching locked dependencies:

```powershell
$repo = 'C:\Users\symbi\source\repos\prism-inference'
$lane = Join-Path ([IO.Path]::GetTempPath()) ('PrismInference-' + [guid]::NewGuid().ToString('N'))
[IO.Directory]::CreateDirectory($lane) | Out-Null
$previous = [Environment]::CurrentDirectory
Push-Location $lane
try {
    [Environment]::CurrentDirectory = $lane
    cargo +nightly-2026-10-22-aarch64-pc-windows-msvc build --locked --offline --release `
        --manifest-path (Join-Path $repo 'Cargo.toml') `
        --target-dir (Join-Path $lane 'build')
    if ($LASTEXITCODE) { throw 'Prism Inference build failed.' }
    & (Join-Path $lane 'build/release/prism-inference.exe') --help
} finally {
    [Environment]::CurrentDirectory = $previous
    Pop-Location
}
```

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

The default `cli` feature also builds `bonsai-token-worker`; `--help` lists its
startup contract. Every setting is required: absolute `--model`, verified
`--bytes`/`--sha256`, `--memory-bytes`, `--context`, `--reserved-output-tokens`,
`--special-input literal|parse`, `--insert-special none|model` and
`--max-frame-bytes` (1 through 1,048,576). Run it from an explicit external
system-temp working directory, using the same pinned toolchain/build commands
above. `cargo run` still selects the original token driver by default.

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

## Memory and ownership

`--memory-bytes` bounds retained model storage, not process memory. Allow additional
headroom for parser/conversion scratch, graph workspace, active state, checkpoints,
optional decoded-vocabulary/output storage and allocator overhead. Text mode admits
at most 1,000,000 vocabulary entries, 64 MiB of decoded piece bytes and 1 MiB per
piece; the entry table and output buffers are additional bounded storage.
Reference sampling admits at most 1,048,576 finite logits and allocates bounded
candidate scratch per draw. BPE admits at most 1,000,000 merge rules and 64 MiB
serialized merge bytes, 64 KiB per pretokenized piece, and 1 MiB/65,536 pieces
per multi-piece request. Vocabulary validation, lookup tables and heap/node/output
scratch add storage outside model/state/checkpoint budgets.

Plain-text requests share the same whole-input byte/piece limits. Special input
admits at most 4,096 control/user-defined spellings, 256 bytes each and 1 MiB in
total. Its sparse trie and per-input match storage are bounded independently;
normal text boundaries borrow the input. The driver drops tokenizer storage
before constructing the graph. These bounds do not qualify process-memory use.

Optional `--state-bytes` sets the admitted maximum
live KV/convolution/recurrent element payload, from 1 byte through the default
4 GiB. `Qwen35::with_state_budget` exposes the same bound to library callers.
Admission charges the complete configured context using FP16 K/V and F32
convolution/recurrent state before state allocation; KV storage remains lazy.
The completion record includes `state_budget_bytes`. This is a payload bound,
not a bound on vector headers, allocator overhead, graph scratch or process RAM;
`--checkpoint-bytes` covers saved snapshots including in-flight rollback.
Process/hardware admission remains the caller's responsibility pending broker
integration.

Graph workspace retains high-water allocations across evaluations/reset. KV
pages grow within the context bound, sealing only complete pages. Checkpoint
admission charges every referenced immutable page per snapshot, page-table
capacity, copied private tails and recurrent storage. This conservatively
overcharges shared payload rather than depending on current reference counts;
the bound covers worst-case independent copies. Tight budgets can reject rollback
capture or retain fewer prefixes. RoPE reuse lasts one
evaluation attempt. Neither retained-model nor checkpoint budgets cover workspace.
Synthetic capacity reuse does not establish full-model allocation, peak-memory or
speed benefit; [R4.2](https://github.com/symbiogenesis/SwiftBonsai2/blob/370084c530e2271f10a1c55a5a10bba7c806aacb/docs/native-rust-engine.md#ordered-work) remains open.

## SIMD and validation

[SIMD S0–S6](https://github.com/symbiogenesis/SwiftBonsai2/blob/370084c530e2271f10a1c55a5a10bba7c806aacb/docs/simd-plan.md) owns kernel status and acceptance.
`MatvecBackend::Auto` selects baseline NEON for complete PQ2/PTQ1 rows when built
for Windows ARM64 MSVC with `experimental-neon` and ambient NEON. PQ2 widths
divisible by 256 retain Q8_K; odd-group PQ2 and PTQ1 retain four Q8_0 scales per
128 weights. Unsupported cases fall back to scalar, which also runs natively
on ARM64. Force it with `MatvecScratch::with_backend(MatvecBackend::Scalar)`;
`selected_backend()` reports actual selection.

Use `Qwen35::with_backend` or the token driver's `--backend auto|scalar|neon`
to apply that policy to the graph's retained projection scratch. The completion
record's `backend_requested` reports the requested policy. It does not claim
every projection used that backend; unsupported operations still use scalar.
`backend_last_operation` reports only the latest matrix attempt: actual selection,
build eligibility, scalar fallback reason, dispatch and success flags. Fallback
reasons distinguish explicit scalar selection, unsupported tensor kinds, disabled
NEON features and unsupported targets. `MatvecScratch::diagnostics` and
`Qwen35::backend_diagnostics` expose the same record. Reset, restore and graph
rejection before matrix preparation leave it unchanged; it can describe
provisional work. A final F32 output projection reports scalar even when earlier
packed projections used NEON.

`backend_statistics` reports cumulative preparation attempts/failures, matrix
attempts/successes, rejection before dispatch, scalar/NEON dispatches and each
scalar fallback reason. `MatvecScratch::statistics` and
`Qwen35::backend_statistics` expose these counts. Prepared siblings share a
preparation and each count as a matrix attempt. Counts include failed/provisional
work and survive graph reset/restore; they are not committed-token accounting.
Counters saturate at `u64::MAX` and retain no input data.

Target-only `fearless_simd` 1.1.0 uses safe core APIs under unchanged
`forbid(unsafe_code)`. Direct packed decode preserves scalar quantization and
ordered scales. Optional ISA detection, per-path memory and
performance qualification remain open.

The original [Foundation](https://github.com/symbiogenesis/SwiftBonsai2/blob/370084c530e2271f10a1c55a5a10bba7c806aacb/.github/workflows/foundation.yml) workflow
established the complete synthetic native suite, formatting, Release
Clippy/rustdoc, feature variants and non-ARM fallback checks. Those checks and
fixtures accompany the extracted runtime. Fixtures cover bounded loading, exact codecs/rounding, graph
state, output reuse, checkpoint isolation and transactional recovery. They do not
load the full model. Locally run focused named regressions, from the same external
working directory convention, for example:

```powershell
cargo +nightly-2026-10-22-aarch64-pc-windows-msvc test --locked --offline --release `
    --manifest-path <repo>/Cargo.toml --target-dir <temp>/build `
    --test runtime edited_turns_restore_complete_recurrent_prefix_and_failed_stream_rolls_back -- --exact
```

The [R0–R7 roadmap](https://github.com/symbiogenesis/SwiftBonsai2/blob/370084c530e2271f10a1c55a5a10bba7c806aacb/docs/native-rust-engine.md) still requires profiling,
full-model logits/states/tokens, complete tokenizer/template and Prism sampler/RNG parity,
adapter execution, broker/OS ownership, broader SIMD/Adreno support and device
correctness/TPS/quality/security acceptance. Full suites and every benchmark
remain CI-only under [validation policy](https://github.com/symbiogenesis/SwiftBonsai2/blob/370084c530e2271f10a1c55a5a10bba7c806aacb/docs/validation.md).

The [upstream Prism baseline](https://github.com/symbiogenesis/SwiftBonsai2/blob/370084c530e2271f10a1c55a5a10bba7c806aacb/docs/prism-upstream-tests.md) separately
imports pinned original C++ tests and applies shared assertions to this runtime.
Its complete source inventory retains unported tests as unsupported, counts them
against the green fraction and prevents already-green adapters from regressing.
