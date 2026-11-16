# Runtime contracts

[CLI and workers](cli-and-workers.md) covers commands and transport.
[Building and validation](build-and-validation.md) covers correctness procedures.

## Status and integration

Prism Inference is an experimental safe Rust CPU runtime for Prism-compatible
GGUF models. Its implemented generation path is dense Qwen35 for pinned
Swift/Bonsai. Broader architectures/model selection and full-model/device
qualification remain open. It links no C/C++ inference library.

The Rust import is `bonsai_prism_runtime`. Executables are `prism-inference`,
`bonsai-token-worker`, `bonsai-swift-worker`, `prism-inference-server` and
`bonsai-gguf`. `SwiftChatSession` supplies verified chat rendering/tokenization;
`SwiftSession` adds MT19937 sampling, five native stop tokens, transactional
decoding and bounded UTF-8 output. Worker clients verify readiness; their owners
control process lifetime and shutdown. The private HTTP server requires an
owning launcher and authenticated prepared-input binding.

## Implemented behavior

### Loading and kernels

- GGUF loading verifies length/SHA-256 before parsing and uses the same retained
  handle. Windows denies write/delete sharing; other platforms require
  caller-enforced immutability. On Windows, artifacts at or above 4 MiB use
  read-only whole-file mapping (`bonsai_native_hash::mmap::MappedReadOnlyFile`
  with 16 MiB `PrefetchVirtualMemory` windows during 4 MiB chunked CNG SHA-256
  hashing), retaining zero-copy `TensorBytes::Mapped(MappedSlice)` storage and
  validating executable tensor scales/extents in parallel across up to 8 workers
  before publishing the immutable `Model` (falling back to buffered reads and
  staging below 4 MiB or on non-Windows targets).
- F32/F16/BF16/PQ2/PTQ1 tensors have checked dimensions/storage budgets. Optional
  identical rank-one B sharing is separate from
  [adapter execution](rank-one-adapter.md).
- Packed scalar kernels preserve PQ2 Q8_K and PTQ1 Q8_0 activation contracts.
  Eligible Windows ARM64 operations can use NEON; unsupported cases fall back
  to scalar. No expanded float matrix persists.
- The graph implements embeddings, sign/Hadamard transforms, RMS, causal
  nonflash attention, interleaved MRoPE, GDN/convolution, gated FFN and output
  projection. FP16 query/probability conversion is preserved. Generic graphs
  default to FP16 KV; Swift uses [K4/V4](quantized-kv.md). Recurrent/convolution
  state remains F32.
- Prepared sibling projections share activation work only under identical
  input/transform/width/contracts. [Execution reference](inference-efficiency.md)
  defines packed layouts, sampling proofs and attention/GDN invariants.

The graph/quantizer reference is Prism
`73bfe6cc485554822f80c473a3c0c4f25e80f41b`; the source-overlay lock is separate.
Audited paths/hashes are in [kernels.rs](../src/kernels.rs), attribution in
[NOTICE](../NOTICE), [PRISM-LICENSE.txt](../PRISM-LICENSE.txt) and
[CAMELID-LICENSE.txt](../CAMELID-LICENSE.txt).

### Transactions and output

Runtime has a single bounded conversation owner. Streaming/backpressure,
edited prompts, cancellation, deadlines and evaluation/delivery failure share a
request transaction. Tokens remain provisional until success. Failed restore or
decoder reset preserves the public committed prefix and closes admission until
reset succeeds; successful reset clears history and saved prefixes.

`Runtime::generate_until` uses an absolute monotonic deadline checked around
decoder operations/callbacks. Synchronous work must return before expiry can be
observed, and rollback completes before return. Caller-owned sampler state rolls
back with decoding. Public reference sampling supports seeded temperature/top-k/
top-p, stable token-ID ties and SplitMix64; Swift uses its pinned native sampler.

`Decoder::evaluate_into` reuses caller logits; `evaluate` remains available.
Shape/finiteness failures clear direct output. Graphs stage per-layer deltas.
`Decoder::ingest` validates intermediate prompt state while omitting output-only
work; its compatibility fallback evaluates/validates/discards logits.
`ingest_block` accepts 1..256 reference tokens and checks cancellation between
them. `generate_with_prefill` / `generate_until_with_prefill` select
`Prefill { block_tokens }`; ordinary calls use one. Explicit
[CPU tiles](cpu-tiled-prefill.md) have separate scratch/numerical identities.

Validated model EOS/EOT/EOM stops combine with caller stops. BOS/padding IDs and
insertion flags are checked; raw-ID prompts receive no automatic insertion.
Bounded incremental output preserves exact token bytes across UTF-8 boundaries,
leaves control-token spellings literal, and flushes any incomplete UTF-8 lead
sequence as `U+FFFD` via `Utf8Output::flush_incomplete` when
`Vocabulary::is_control` identifies a control boundary.

Worker and server cancellation distinguish persistent pipes from batch inputs:
`bonsai-swift-worker` reads stdin asynchronously and activates `Cancellation`
immediately upon stdin EOF or read failure without blocking on a stalled stdin
thread at exit, whereas `bonsai-token-worker` drains buffered stdin chunks
through a bounded channel reader (`sync_channel(4)`) before EOF. The private
HTTP backend supports `Engine::prepare_until` for cancellation and deadlines
during prompt preparation and bounds post-request recovery via
`Worker::settled_within` (`WORKER_SETTLE_TIMEOUT = 10s`, returning
`NativeBackendError::Deadline` on timeout).

### Checkpoints

Full-attention checkpoints share sealed immutable 64-token pages and copy private
tails plus complete F32 recurrent/convolution state. Subsequent append/reset/
restore cannot mutate snapshots. Immutable complete logits share an `Arc`;
evaluation detaches shared output via `Arc::try_unwrap` without copying obsolete
values, then reuses unique logits scratch alongside `Decoder::checkpoint_into`
and `Runtime`'s bounded 2-slot recycled checkpoint pool. Checkpoints never alias
mutable graph scratch.

Position-zero checkpoints retain provenance/metadata instead of canonical empty
KV and positive-zero recurrent payload. Restore validates complete geometry,
position alignment, `KvCache::same_representation` /
`PackedCache::same_representation`, packed page/tail partitions, and
`ProjectedQwen35`'s SHA-256 `projection_plan_digest` (including per-layer active
LoRA FFN fusion via `adapter_affects_any`: keeping `external_ffn_chain` when
`[ffn_gate, ffn_up, ffn_down]` have no active LoRA and `external_ffn_swiglu` when
`[ffn_gate, ffn_up]` are untargeted or inactive) before clearing or mutating
live state; `Qwen35::restore` clears `frequencies_ready` cleanly. Nonempty
snapshots remain complete. `Qwen35` tracks `state_consistent` across
`ingest_cpu_tile` and rejects subsequent decode/prefill/checkpoint calls after
any mid-tile error until `reset` or valid `restore` succeeds.
`CausalDepthwiseConv1d` performs atomic overlapping shifts (`steps < 3`, `== 3`,
`> 3`) without corrupting its 3-tap logical history on nonfinite error, and
`recurrent_decay` rejects positive `ssm_a > 0.0` at construction while guarding
`alpha + time_bias >= 88.0` overflow (`decay == 1.0` when `ssm_a == 0.0`). Reset
skips already empty state; scale reset still renews identity/provenance.

Transient verification/rollback snapshots recycle without inserting into or
evicting reusable prefix cache entries. Opt-in
`PromptReuse::PromptEndAndLatest` retains the latest prompt-end checkpoint across
assistant output; `HybridCheckpointBoundary::reconcile` computes the maximal
shared boundary across token history, KV page alignment (or exact unaligned KV
bounds), non-truncatable recurrent GDN/conv state and representation/adapter
identity (`KvCache::truncate_to` and `PackedCache::truncate_to` stay pure-KV).
Optional prefix replacement preserves older prefixes if capture/replacement
fails. Count eviction follows successful capture; byte-pressure eviction precedes
allocation to honor the hard bound. [Session profiles](swift-session-profiles.md)
define adapter-scale provenance, latest reuse and reset-only terminal ingestion.

### Timing

`Runtime::generate_sampled_until_with_timings` and
`SwiftSession::complete_with_timings` opt into fixed-size attribution; ordinary
entry points make no timing reads. Phases cover sampler/rollback snapshots,
prefix preparation, state-only prefill, prompt/decode evaluation, sampling,
callbacks, optional prefix capture and error restore. Reset-only sampled service
adds [terminal-ingestion](swift-session-profiles.md#terminal-state-only-ingestion)
after the eleven existing indexes.

Measurements include checks/allocation rather than isolated kernels. Failed
phases retain completed lower bounds and unknown partial work. Stop draws have
no decode/callback work. Runtime success and final session delivery are separate;
a missing receipt establishes no token count.

### Tokenization and codecs

Tokenization supports bounded GPT-2/Qwen35 vocabularies, the pinned Qwen35 scanner/
Unicode tables, explicit literal/parse special handling and ranked BPE.
Pretokenized BPE preserves piece boundaries and emits normal IDs only.
Model-declared BOS/EOS insertion is explicit. Template probing is separate from
full-input chat/tokenizer qualification.

`TokenizationSession` verifies once, retains tokenizer/special declarations and
shares context/output accounting across text/raw IDs; it releases metadata and
the file handle after construction. `TextOutputSession` similarly retains a
verified output vocabulary. `Utf8Output` keeps at most three incomplete bytes
and resets abandoned output.

The token worker serves bounded JSONL without graph allocation or further model
I/O. `TokenClient` verifies pinned readiness, framing/schema/correlation and
fails closed after transport faults. Library callers share bounded prompt/JSON
readers and checked stop policy with the CLI.

`jinja::Template::chat_capabilities` probes tool definitions/calls, roles,
parallel calls and typed/string content using pinned markers and fallback rules.
`codec::CodecBuffers` supplies bounded regions, monotonic handles and explicit
reference/CPU/decode/dot calls. Safe byte access permits unaligned/overlapping
regions; failure preserves output. Its 144-slot inventory distinguishes native
availability from implemented Rust operations. Unavailable CPU quantizers reject.

### Private context rejection

Only concrete `TokenLimitExceeded`, produced under validated session context/
output reservation, maps to HTTP `ContextLimit`. Tokenizer/schema/byte/piece/
allocation failures retain their classifications; display text never selects
status. Pre-generation overflow returns `413 Content Too Large` with exactly
`{"error":{"code":"context_length_exceeded"}}`, bounded length, close/no-store/
nosniff headers. Rejection invalidates the counted seal, performs no generation
and preserves a healthy owner; it does not reset a faulted generation.

## Memory and ownership

`--memory-bytes` bounds retained model storage, not process RAM. Add parser/
conversion scratch, workspace, live state, checkpoints, output/vocabulary,
allocator and process/device overhead. Owners enforce process/hardware admission.

Swift context admission uses immutable selected KV policy before artifact I/O:

| Policy | Per-token process allowance | Maximum admitted context |
| --- | ---: | ---: |
| FP16 | 131,296 bytes | 62,720 |
| K4/V4 | 34,016 bytes | 262,144 |
| K4/V3 | 29,920 bytes | 262,144 |

Per-token charges include two full caches/scales, 96 bytes of sequential scores
and 128 bytes of page/token metadata. The fixed allowance is 11 GiB.
`validate_context_capacity_with_kv` and prefill/execution variants bind policy;
compatibility validators retain FP16. Generic/FP16/K4/V3 state ceilings are
4 GiB. Full-context K4/V4 requires the narrow exact Swift PQ2/PTQ1 constructor
and pinned 16-attention/48-recurrent geometry, admitting 4,644,798,464 bytes.
Individual allocation/scratch limits remain in force. Live/rollback reserves
scale with payload with a 1 GiB floor; speculation adds another complete prefix
and verification scratch. Shared pages do not discount admission.

`SwiftExecutionMemoryLedger` (`SwiftExecutionMemoryLedger::for_swift_profile`,
`SwiftSession::execution_memory_ledger`) separates Windows Job-object commit
components (`job_commit_total_bytes`: pinned host weights, live/checkpointed
recurrent and KV state, verification logits, replay scratch, tile scratch and
fragmentation margin) from `accelerator_external_bytes` (added only in
`total_with_external_bytes`). `swift_worker_request` also bounds initial
message `Vec` preallocation to 64 entries before enforcing `MAX_MESSAGES`.

Private startup validates paired `--context-tokens`/`--memory-limit-bytes`
against final cache selection regardless of argument order. Context must be at
least 4,352 tokens and a multiple of 256, within the selected cache ceiling.
The owner enforces the process allowance and separately retains a 4 GiB physical
host reserve. Trusted
`--placement auto|cpu` is immutable: CPU skips package discovery/accelerator
initialization; `native_backend::serve_with_placement` clears a supplied executor.
CPU readiness adds `placement: "cpu"` and `projection_backend: "cpu"`.
This selection does not lower fixed/process/state reserves.

`--state-bytes` / `Qwen35::with_state_budget` bound configured full-context
KV/recurrent/convolution element payload before lazy KV allocation; the CLI
reports `state_budget_bytes`. `--checkpoint-bytes` separately bounds saved/
in-flight rollback snapshots. Tables, allocator overhead and graph workspace
are additional.

Checkpoint admission charges every shared page per owner, page-table capacity,
private tails, recurrent payload and full logits capacity/header/Arc counters.
Empty snapshots charge conservative metadata but no canonical-zero payload or
spare reset KV. Workspace retains high-water capacity across reset. Independent
checkpoint ownership and conservative charging do not imply measured peak RAM.

Other bounded storage:

| Area | Limits |
| --- | --- |
| Decoded output vocabulary | 1,000,000 entries; 64 MiB total piece bytes; 1 MiB/piece |
| Reference sampling | 1,048,576 finite logits plus candidate scratch |
| Swift sampling | 248,320 logits; bounded top-20 candidate heap |
| BPE | 1,000,000 rules; 64 MiB serialized rules; 64 KiB/pretokenized piece |
| Multi-piece request | 1 MiB / 65,536 pieces |
| Special input | 4,096 spellings; 256 bytes each; 1 MiB total |

Lookup tables, candidate/output buffers and allocator overhead are additional.
The driver drops tokenizer storage before graph construction.
[Bounded request preparation](swift-session-profiles.md#bounded-request-preparation)
defines special matching scratch.

## SIMD dispatch and diagnostics

`MatvecBackend::Auto` selects baseline NEON for eligible packed rows on Windows
ARM64 MSVC with `experimental-neon` and ambient NEON. PQ2 widths divisible by
256 use Q8_K; odd-group PQ2/PTQ1 use Q8_0. Unsupported cases run scalar, including
natively on ARM64. Force policy through `MatvecScratch::with_backend`,
`Qwen35::with_backend` or CLI `--backend auto|scalar|neon`.

`selected_backend()` reports selection. Completion `backend_requested` reports
policy; `backend_last_operation` reports only the latest matrix attempt's actual
selection, eligibility, fallback reason, dispatch and success. Graph rejection,
reset/restore leave that record unchanged. It can describe provisional work;
a final F32 projection can report scalar after packed NEON projections.

`MatvecScratch::diagnostics/statistics` and
`Qwen35::backend_diagnostics/backend_statistics` expose the same records.
Cumulative counters include preparations/failures, matrix attempts/successes,
pre-dispatch rejection, dispatches and fallback reasons. Siblings share one
preparation but count separate matrices. Counts survive reset/restore, saturate
at `u64::MAX`, retain no inputs and do not count committed tokens.

`Qwen35::fusion_statistics` separately witnesses the existing decode
RMSNorm/Hadamard, residual/RMSNorm/Hadamard, SwiGLU/Hadamard, adapter-preserving
SwiGLU/Hadamard and external FFN branches. Each path counts attempts, complete
successful returns and explicit fallback. A successful separate fallback never
counts as a completed fusion. Counts survive reset/restore and describe
provisional execution; they establish neither hardware execution, tiled-prefill
coverage nor numerical or performance qualification.

The model-free `fusion_verification_` controls compare real graph preparation
branches with separate operations, checking output bits and reporting maximum
absolute error on failure alongside the executed counts. They cover source and
pinned arithmetic, changed unrelated transform entries, explicit worker budgets
and a failing fused preparation followed by successful ordinary normalization.
The existing adapter FFN controls additionally require a completed fusion witness.
The methodology follows ggml's [fusion testing][fusion-tests] and
[boundary-dependent rounding regression][fusion-boundaries]; no upstream
implementation is imported.

Prepared external FFN-chain, FFN-SwiGLU and sibling-group calls validate their
specific F32 capability before submission, including direct library calls.
Unsupported capabilities reject and clear provisional outputs without calling
the executor. This does not substitute F32 arithmetic for the pinned F16/BF16
activation-rounding contracts. `projection_precision_` controls distinguish
F32 activation bits, BF16 range beyond F16, intentional BF16 rounding and F16
range rejection, following the [operand-precision regression approach][precision-tests].

[fusion-tests]: https://github.com/ggml-org/llama.cpp/pull/30769
[fusion-boundaries]: https://github.com/ggml-org/llama.cpp/pull/31688
[precision-tests]: https://github.com/ggml-org/llama.cpp/pull/31896
