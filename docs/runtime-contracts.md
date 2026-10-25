# Runtime contracts

[Documentation index](index.md) · [CLI and workers](cli-and-workers.md) · [Roadmap](roadmap.md)

## Status and integration

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
are in [kernels.rs](../src/kernels.rs); MIT terms are in [NOTICE](../NOTICE) and
[PRISM-LICENSE.txt](../PRISM-LICENSE.txt). HTTP framing and Q1_0/Q2_0 decoder
adaptations also retain Camelid's MIT terms in
[CAMELID-LICENSE.txt](../CAMELID-LICENSE.txt).

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
Process/hardware admission remains the caller's responsibility. SwiftBonsai
supplies its guarded development launcher; these payload bounds do not admit
process memory.

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

## SIMD dispatch and diagnostics

The inherited [SIMD S0–S6 plan](https://github.com/symbiogenesis/SwiftBonsai2/blob/370084c530e2271f10a1c55a5a10bba7c806aacb/docs/simd-plan.md) records kernel status and remaining acceptance.
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
