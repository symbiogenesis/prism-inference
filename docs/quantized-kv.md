# Experimental Snapdragon rotated KV

This source candidate adds a clean-room Rust cache codec for the 256-channel
full-attention heads used by Swift. The public graph policy is
`graph::KvCachePolicy::{F16, Rotated4, Rotated4Keys3Values}`. Existing callers and
serving defaults retain FP16. No released Runner bundle selects this candidate.

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
The graph's existing projection worker policy is preserved; compressed attention
currently streams query heads serially. Further GQA head tiling requires measured
benefit and its own correctness checks.

Queries rotate once per head. Key scoring reads packed codes directly. Weighted
values accumulate in the rotated basis before one inverse transform. Attention
never allocates a context-sized expanded key/value buffer. Current-token K/V use
the same codec as cached history. Existing query/key RMS normalization, RoPE and
FP16 preparation are retained; the pinned Swift softmax arithmetic has a separate
262,144-token bound for this profile. Ordinary FP16 softmax keeps its 8K bound.

## Selection, identity and recovery

The raw generation CLI accepts `--kv-cache f16|rsq4|rsq4k3v`. It rejects that option
in tokenize mode. Completion metadata reports `kv_cache` and `graph_identity`.
`Qwen35::with_state_budget_and_kv` admits full-context payload including scales;
`with_rank_one_adapter_and_kv` preserves explicit adapter selection and scale.

`SwiftSession::load_profile_with_kv_cache` and the private server's trusted
`--kv-cache` option expose the candidate at the existing fixed 8K service bound.
Client requests cannot change precision. Non-FP16 readiness adds `kv_cache` and
`startup_graph_identity`; the latter explicitly identifies construction before
any request changes the adapter scale. Old strict Runner clients reject that
different profile. Application integration requires an owner-published verified
runtime bundle and a matching broker capability contract; no active pin is changed.

Packed caches seal immutable 64-token pages and copy only their bounded tails
when checkpointing. Every row is encoded and all storage growth is reserved before
the graph commits any KV, GDN, convolution or position delta. Snapshots restore
the exact selected packed representation, not the original FP16 approximation.
F32 GDN/convolution state stays unchanged. Codec/profile identities prevent
cross-profile checkpoint reuse. Tiled prefill is explicitly rejected for these
profiles; ordinary cancellation-aware sequential block ingestion remains supported.

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

Default or mandatory adoption is conditional on predeclared, owner-accepted
negligible-quality-loss thresholds and a useful measured capacity or latency benefit.
Retain losing modes and counterexamples. Qualify X1 and X2 independently with exact
source/model/adapter/profile identities and matched power observations. Complete
suites and benchmarks remain owner-operated CI. Bounded laptop work retains the
aggregate budget and power rules in `AGENTS.md`; no historical allowance reopens.
