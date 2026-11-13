# Development-default Snapdragon rotated KV

This source candidate adds a clean-room Rust cache codec for the 256-channel
full-attention heads used by Swift. The public graph policy is
`graph::KvCachePolicy::{F16, Rotated4, Rotated4FittedValues, Rotated4Keys3Values}`. K4/V4 (`rsq4`) is
the owner-selected development default for Swift generation, sessions and workers.
It is required within that selected service profile; request bodies cannot change
precision and failures never switch to FP16. Generic graph constructors retain
FP16 for reference controls. No released Runner bundle selects this source yet.

This development choice does not require identical tokens, FP16 logits or perfect
reconstruction. Actual-model quality, memory benefit, responsiveness and separate
X1/X2 device qualification remain pending; they are not claimed by the default.

## Algorithm and CPU implementation

The codec combines three deterministic signed, normalized Walsh-Hadamard passes
with symmetric Gaussian Lloyd-Max scalar levels. Four-bit records occupy 132
bytes per head, including a four-byte little-endian F32 norm scale; three-bit
records occupy 100 bytes. Packed codes use contiguous least-significant bits
first. Zero has one canonical representation. Seeds, tables, packing and rounding
belong to version 1 of the codec; the sampler seed does not select cache rotations.

This is an original implementation of mathematical ideas from
[TurboQuant](https://arxiv.org/abs/2504.19874), informed by
[Qdrant's CPU implementation discussion](https://qdrant.tech/articles/turboquant-quantization/)
and [turbovec's compressed scoring design](https://github.com/RyanCodrai/turbovec).
It imports no code or runtime dependency from those projects. It is not Google's
implementation, does not implement QJL, and has no Google or Qdrant endorsement.
Gaussian levels are an approximation to the rotated 256-dimensional distribution;
neither unbiased attention nor negligible downstream quality loss is claimed.

The safe codec uses a narrow audited `bonsai-neon-dot` boundary for runtime-detected
128-bit NEON. X1 Elite and X2 Elite Extreme share these kernels and the same
numerical identity. Explicit scalar construction supplies an independent fallback.
No unmeasured SKU-specific thread count, optional ISA or speed advantage is assumed.
New local source shares the graph's existing private row-worker pool across
independent compressed-attention query heads. Each head retains its exact
score, softmax, value accumulation and inverse-transform order. Tasks own
disjoint attended/score slices, borrow immutable packed/current records and join
before returning the earliest head-chunk failure. Serial and feature-disabled
paths retain their existing policy; no second pool or expanded cache is added.
Cancellation retains the ordinary synchronous-token boundary. This source
candidate is separate from the published 0.1.25 serial control and has no
measured device or complete-request benefit.

Queries rotate once per head. Key scoring reads packed codes directly. Weighted
values accumulate in the rotated basis before one inverse transform. Attention
never allocates a context-sized expanded key/value buffer. Current-token K/V use
the same codec as cached history. Existing query/key RMS normalization, RoPE and
FP16 preparation are retained; the pinned Swift softmax arithmetic has a separate
262,144-token bound for this profile. Ordinary FP16 softmax keeps its 8K bound.

## Selection, identity and recovery

The raw generation CLI accepts `--kv-cache f16|rsq4|rsq4vf|rsq4k3v`, defaulting to `rsq4`
only after loading verifies the exact Swift PQ2/PTQ1 identity. Other models retain
FP16. Tokenize mode rejects this option. Explicit `f16` supplies comparison/recovery;
unsupported rotated geometry rejects instead of falling back. Completion metadata reports `kv_cache`
and `graph_identity`. K4/V3 remains a separate opt-in capacity experiment.
`Qwen35::with_state_budget_and_kv` admits full-context payload including scales;
`with_rank_one_adapter_and_kv` preserves explicit adapter selection and scale.

Ordinary `SwiftSession` loading uses `DEFAULT_KV_CACHE` (`Rotated4`).
`SwiftSession::load_profile_with_kv_cache` and both workers' trusted `--kv-cache`
options supply explicit FP16 comparison/recovery at the existing fixed 8K bound.
Client requests cannot change precision. Non-FP16 readiness adds `kv_cache` and
`startup_graph_identity`; the latter explicitly identifies construction before
any request changes the adapter scale. Old strict Runner clients reject that
different profile. Application integration requires an owner-published verified
runtime bundle and a matching broker capability contract; no active pin is changed.
The JSONL `SwiftClient::connect` verifies the default cache and exact startup graph
identity. Use `connect_with_kv_cache` when explicitly launching a different cache;
FP16 retains its original readiness shape. Restart at a fresh-session boundary
to change the immutable precision policy.

### Guarded fitted-value scale candidate

Explicit `rsq4vf` (`Rotated4FittedValues`) keeps version-1 four-bit keys and
fits only the scales of four-bit values. It is available through the generation
CLI, JSONL worker, private server and library selection above. It does not change
`DEFAULT_KV_CACHE`, Runner's published runtime or request-time precision policy.
The candidate retains the ordinary 4 GiB graph state ceiling; its session capacity
calculation admits at most 241,664 tokens, without qualifying that context.

`Quantizer::with_fitted_scale` selects codec version 2. After version-1 normalization,
rotation and code selection, it fits `s = s_norm * dot(y,c) / dot(c,c)`, where `y`
is the normalized rotated row and `c` contains the selected centroids. Both sums
use ordered binary64 fused multiply-add operations; the proposed scale rounds
to binary32. Nonpositive, nonfinite or reconstruction-overflowing candidates
retain the original scale. Codes are never reassigned.

Acceptance requires strictly lower measured squared error against
`binary64(s_norm) * binary64(y[i])`, with actual binary32 `s*c[i]` reconstruction
products and ordered binary64 squared-error accumulation. Equal error retains
the baseline, including changes lost to rounding. This guard precedes the rounded
inverse transform: it does not guarantee lower original-space error, attention
output error, logits error or better answers. Canonical zero and atomic failures
retain their existing contracts.

K and V records still occupy 132 bytes per head, including the same four-byte
scale. Readers, packed attention arithmetic, codebooks, rotations and retained
scratch are unchanged. Encoding adds two bounded 256-coordinate scans and a
division per nonzero value head; complete-request cost remains unmeasured.
Current-token values use the same fitter as retained history. For identical input
projections, key records and attention probabilities remain unchanged; future
model states may diverge through changed values.

The graph identity component `kv-rsq4vf-d256-rht3-v1` binds this policy's version-1
keys and version-2 value scales; `rht3-v1` retains the rotation/layout family name.
Checkpoint copies preserve the quantizer, and graph/prefix/readiness identities
reject reuse across `rsq4` and `rsq4vf`. No record bytes are used for identity.
Focused numerical fixtures cannot qualify native encoding cost or held-out model
quality. Compare those before selecting this candidate as a default, retaining
losing results and the existing power, budget and promotion requirements.

Packed caches seal immutable 64-token pages and copy only their bounded tails
when checkpointing. Every row is encoded and all storage growth is reserved before
the graph commits any KV, GDN, convolution or position delta. Snapshots restore
the exact selected packed representation, not the original FP16 approximation.
F32 GDN/convolution state stays unchanged. Codec/profile identities prevent
cross-profile checkpoint reuse. New local CPU tiles share packed layer/output
projections while retaining causal per-token quantized attention and encoded
append. Their target verifier supplies per-position logits at the selected cache
precision. No FP16 history expansion or additional worker pool is allocated.
Ordinary sequential ingestion remains supported; triangular GDN stays a separate
FP16 profile. These consumers require their own source-bound release and gates.

## Memory observations

`Qwen35::state_memory` and `SwiftSession::state_memory` observe committed KV
payload (including packed scales), the configured full-context payload bound,
unchanged FP32 recurrent/convolution payload and live state storage capacities.
The generation CLI emits these fields as `state_memory` in its completion record,
separately from the configured `state_budget_bytes`. Observation allocates no
expanded cache and does not change state or checkpoint ownership.

Live storage charges layer tables, cache metadata and retained buffer capacities,
including buffers left after reset; referenced shared pages are charged in full.
It excludes model weights, workspace scratch, external retained checkpoints,
allocator overhead and process/device allocations. The full-context field is a
calculation rather than an allocation observation. Use independent complete-task
process/Job/residency measurements to establish an application memory benefit.
A zero committed payload after reset does not establish that all memory was freed.

## Memory and promotion

For 16 full-attention layers and four KV heads, maximum packed K/V payload is:

| Total context | FP16 | K4/V4 | K4/V3 |
| --- | ---: | ---: | ---: |
| 8,192 | 512 MiB | 132 MiB | 116 MiB |
| 32,768 | 2,048 MiB | 528 MiB | 464 MiB |
| 65,536 | 4,096 MiB | 1,056 MiB | 928 MiB |

These are calculations, not process-memory observations. Add weights, F32 recurrent
state, page tables, codec sign masks, bounded preparation/attention scratch,
snapshots, allocator overhead and the existing physical reserve. Nothing increases
the active application's context limit or reduces its admission policy automatically.

Required independent acceptance includes scalar/NEON fixture equality, packed
layout and overflow controls, independent decoded attention/GQA comparisons,
checkpoint page boundaries, cancellation and failed-delivery recovery. These
model-free checks do not establish long-context retrieval, complete-answer quality,
actual-model trajectories, adapter quality or device speed.

The owner has selected K4/V4 as the development default before full qualification.
Keep the predeclared practical quality/memory packet for a measured benefit and
device verdict; source selection cannot pass it. Retain losing modes and
counterexamples. Qualify X1 and X2 independently with exact
source/model/adapter/profile identities and matched power observations. Complete
suites and benchmarks remain owner-operated CI. Bounded laptop work retains the
aggregate budget and power rules in `AGENTS.md`; no historical allowance reopens.
